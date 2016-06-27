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

#include <dlfcn.h>

const char * tau_orig_libname = "tau_shmem";
static void *tau_handle = NULL;


static void * get_function_handle(char const * name)
{
  char const * err;
  void * handle;

  // Reset error pointer
  dlerror();

  fprintf(stderr, "Using dlsym to get %s\n", name); fflush(stdout);

  // Attempt to get the function handle
  handle = dlsym(RTLD_NEXT, name);

  // Detect errors
  if ((err = dlerror())) {
    // These calls are unsafe, but we're about to die anyway.     
    fprintf(stderr, "Error getting %s handle: %s\n", name, err);  
    fflush(stderr);
    exit(1);
  }

  return handle;
}

/**********************************************************
   shmem_short_get
 **********************************************************/

extern void  __wrap_shmem_short_get(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __real_shmem_short_get(short * a1, const short * a2, size_t a3, int a4)  {

  typedef void (*shmem_short_get_t)(short * a1, const short * a2, size_t a3, int a4);
  shmem_short_get_t shmem_short_get_handle = (shmem_short_get_t)get_function_handle("shmem_short_get");
  shmem_short_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_short_get(short * a1, const short * a2, size_t a3, int a4) {
   __wrap_shmem_short_get(a1, a2, a3, a4);
}

extern void shmem_short_get_(short * a1, const short * a2, size_t a3, int a4)
{
   __wrap_shmem_short_get(a1, a2, a3, a4);
}

extern void shmem_short_get__(short * a1, const short * a2, size_t a3, int a4)
{
   __wrap_shmem_short_get(a1, a2, a3, a4);
}

extern void SHMEM_SHORT_GET_(short * a1, const short * a2, size_t a3, int a4)
{
   __wrap_shmem_short_get(a1, a2, a3, a4);
}

extern void SHMEM_SHORT_GET__(short * a1, const short * a2, size_t a3, int a4)
{
   __wrap_shmem_short_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_get
 **********************************************************/

extern void  __wrap_shmem_int_get(int * a1, const int * a2, size_t a3, int a4) ;
extern void  __real_shmem_int_get(int * a1, const int * a2, size_t a3, int a4)  {

  typedef void (*shmem_int_get_t)(int * a1, const int * a2, size_t a3, int a4);
  shmem_int_get_t shmem_int_get_handle = (shmem_int_get_t)get_function_handle("shmem_int_get");
  shmem_int_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_int_get(int * a1, const int * a2, size_t a3, int a4) {
   __wrap_shmem_int_get(a1, a2, a3, a4);
}

extern void shmem_int_get_(int * a1, const int * a2, size_t a3, int a4)
{
   __wrap_shmem_int_get(a1, a2, a3, a4);
}

extern void shmem_int_get__(int * a1, const int * a2, size_t a3, int a4)
{
   __wrap_shmem_int_get(a1, a2, a3, a4);
}

extern void SHMEM_INT_GET_(int * a1, const int * a2, size_t a3, int a4)
{
   __wrap_shmem_int_get(a1, a2, a3, a4);
}

extern void SHMEM_INT_GET__(int * a1, const int * a2, size_t a3, int a4)
{
   __wrap_shmem_int_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_get
 **********************************************************/

extern void  __wrap_shmem_long_get(long * a1, const long * a2, size_t a3, int a4) ;
extern void  __real_shmem_long_get(long * a1, const long * a2, size_t a3, int a4)  {

  typedef void (*shmem_long_get_t)(long * a1, const long * a2, size_t a3, int a4);
  shmem_long_get_t shmem_long_get_handle = (shmem_long_get_t)get_function_handle("shmem_long_get");
  shmem_long_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_long_get(long * a1, const long * a2, size_t a3, int a4) {
   __wrap_shmem_long_get(a1, a2, a3, a4);
}

extern void shmem_long_get_(long * a1, const long * a2, size_t a3, int a4)
{
   __wrap_shmem_long_get(a1, a2, a3, a4);
}

extern void shmem_long_get__(long * a1, const long * a2, size_t a3, int a4)
{
   __wrap_shmem_long_get(a1, a2, a3, a4);
}

extern void SHMEM_LONG_GET_(long * a1, const long * a2, size_t a3, int a4)
{
   __wrap_shmem_long_get(a1, a2, a3, a4);
}

extern void SHMEM_LONG_GET__(long * a1, const long * a2, size_t a3, int a4)
{
   __wrap_shmem_long_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_get
 **********************************************************/

extern void  __wrap_shmem_longlong_get(long long * a1, const long long * a2, size_t a3, int a4) ;
extern void  __real_shmem_longlong_get(long long * a1, const long long * a2, size_t a3, int a4)  {

  typedef void (*shmem_longlong_get_t)(long long * a1, const long long * a2, size_t a3, int a4);
  shmem_longlong_get_t shmem_longlong_get_handle = (shmem_longlong_get_t)get_function_handle("shmem_longlong_get");
  shmem_longlong_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_get(long long * a1, const long long * a2, size_t a3, int a4) {
   __wrap_shmem_longlong_get(a1, a2, a3, a4);
}

extern void shmem_longlong_get_(long long * a1, const long long * a2, size_t a3, int a4)
{
   __wrap_shmem_longlong_get(a1, a2, a3, a4);
}

extern void shmem_longlong_get__(long long * a1, const long long * a2, size_t a3, int a4)
{
   __wrap_shmem_longlong_get(a1, a2, a3, a4);
}

extern void SHMEM_LONGLONG_GET_(long long * a1, const long long * a2, size_t a3, int a4)
{
   __wrap_shmem_longlong_get(a1, a2, a3, a4);
}

extern void SHMEM_LONGLONG_GET__(long long * a1, const long long * a2, size_t a3, int a4)
{
   __wrap_shmem_longlong_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float_get
 **********************************************************/

extern void  __wrap_shmem_float_get(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __real_shmem_float_get(float * a1, const float * a2, size_t a3, int a4)  {

  typedef void (*shmem_float_get_t)(float * a1, const float * a2, size_t a3, int a4);
  shmem_float_get_t shmem_float_get_handle = (shmem_float_get_t)get_function_handle("shmem_float_get");
  shmem_float_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float_get(float * a1, const float * a2, size_t a3, int a4) {
   __wrap_shmem_float_get(a1, a2, a3, a4);
}

extern void shmem_float_get_(float * a1, const float * a2, size_t a3, int a4)
{
   __wrap_shmem_float_get(a1, a2, a3, a4);
}

extern void shmem_float_get__(float * a1, const float * a2, size_t a3, int a4)
{
   __wrap_shmem_float_get(a1, a2, a3, a4);
}

extern void SHMEM_FLOAT_GET_(float * a1, const float * a2, size_t a3, int a4)
{
   __wrap_shmem_float_get(a1, a2, a3, a4);
}

extern void SHMEM_FLOAT_GET__(float * a1, const float * a2, size_t a3, int a4)
{
   __wrap_shmem_float_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_double_get
 **********************************************************/

extern void  __wrap_shmem_double_get(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __real_shmem_double_get(double * a1, const double * a2, size_t a3, int a4)  {

  typedef void (*shmem_double_get_t)(double * a1, const double * a2, size_t a3, int a4);
  shmem_double_get_t shmem_double_get_handle = (shmem_double_get_t)get_function_handle("shmem_double_get");
  shmem_double_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_double_get(double * a1, const double * a2, size_t a3, int a4) {
   __wrap_shmem_double_get(a1, a2, a3, a4);
}

extern void shmem_double_get_(double * a1, const double * a2, size_t a3, int a4)
{
   __wrap_shmem_double_get(a1, a2, a3, a4);
}

extern void shmem_double_get__(double * a1, const double * a2, size_t a3, int a4)
{
   __wrap_shmem_double_get(a1, a2, a3, a4);
}

extern void SHMEM_DOUBLE_GET_(double * a1, const double * a2, size_t a3, int a4)
{
   __wrap_shmem_double_get(a1, a2, a3, a4);
}

extern void SHMEM_DOUBLE_GET__(double * a1, const double * a2, size_t a3, int a4)
{
   __wrap_shmem_double_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float128_get
 **********************************************************/

extern void  __wrap_shmem_float128_get(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __real_shmem_float128_get(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  typedef void (*shmem_float128_get_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4);
  shmem_float128_get_t shmem_float128_get_handle = (shmem_float128_get_t)get_function_handle("shmem_float128_get");
  shmem_float128_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float128_get(__float128 * a1, const __float128 * a2, size_t a3, int a4) {
   __wrap_shmem_float128_get(a1, a2, a3, a4);
}

extern void shmem_float128_get_(__float128 * a1, const __float128 * a2, size_t a3, int a4)
{
   __wrap_shmem_float128_get(a1, a2, a3, a4);
}

extern void shmem_float128_get__(__float128 * a1, const __float128 * a2, size_t a3, int a4)
{
   __wrap_shmem_float128_get(a1, a2, a3, a4);
}

extern void SHMEM_FLOAT128_GET_(__float128 * a1, const __float128 * a2, size_t a3, int a4)
{
   __wrap_shmem_float128_get(a1, a2, a3, a4);
}

extern void SHMEM_FLOAT128_GET__(__float128 * a1, const __float128 * a2, size_t a3, int a4)
{
   __wrap_shmem_float128_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_put
 **********************************************************/

extern void  __wrap_shmem_short_put(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __real_shmem_short_put(short * a1, const short * a2, size_t a3, int a4)  {

  typedef void (*shmem_short_put_t)(short * a1, const short * a2, size_t a3, int a4);
  shmem_short_put_t shmem_short_put_handle = (shmem_short_put_t)get_function_handle("shmem_short_put");
  shmem_short_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_short_put(short * a1, const short * a2, size_t a3, int a4) {
   __wrap_shmem_short_put(a1, a2, a3, a4);
}

extern void shmem_short_put_(short * a1, const short * a2, size_t a3, int a4)
{
   __wrap_shmem_short_put(a1, a2, a3, a4);
}

extern void shmem_short_put__(short * a1, const short * a2, size_t a3, int a4)
{
   __wrap_shmem_short_put(a1, a2, a3, a4);
}

extern void SHMEM_SHORT_PUT_(short * a1, const short * a2, size_t a3, int a4)
{
   __wrap_shmem_short_put(a1, a2, a3, a4);
}

extern void SHMEM_SHORT_PUT__(short * a1, const short * a2, size_t a3, int a4)
{
   __wrap_shmem_short_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_put
 **********************************************************/

extern void  __wrap_shmem_int_put(int * a1, const int * a2, size_t a3, int a4) ;
extern void  __real_shmem_int_put(int * a1, const int * a2, size_t a3, int a4)  {

  typedef void (*shmem_int_put_t)(int * a1, const int * a2, size_t a3, int a4);
  shmem_int_put_t shmem_int_put_handle = (shmem_int_put_t)get_function_handle("shmem_int_put");
  shmem_int_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_int_put(int * a1, const int * a2, size_t a3, int a4) {
   __wrap_shmem_int_put(a1, a2, a3, a4);
}

extern void shmem_int_put_(int * a1, const int * a2, size_t a3, int a4)
{
   __wrap_shmem_int_put(a1, a2, a3, a4);
}

extern void shmem_int_put__(int * a1, const int * a2, size_t a3, int a4)
{
   __wrap_shmem_int_put(a1, a2, a3, a4);
}

extern void SHMEM_INT_PUT_(int * a1, const int * a2, size_t a3, int a4)
{
   __wrap_shmem_int_put(a1, a2, a3, a4);
}

extern void SHMEM_INT_PUT__(int * a1, const int * a2, size_t a3, int a4)
{
   __wrap_shmem_int_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_put
 **********************************************************/

extern void  __wrap_shmem_long_put(long * a1, const long * a2, size_t a3, int a4) ;
extern void  __real_shmem_long_put(long * a1, const long * a2, size_t a3, int a4)  {

  typedef void (*shmem_long_put_t)(long * a1, const long * a2, size_t a3, int a4);
  shmem_long_put_t shmem_long_put_handle = (shmem_long_put_t)get_function_handle("shmem_long_put");
  shmem_long_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_long_put(long * a1, const long * a2, size_t a3, int a4) {
   __wrap_shmem_long_put(a1, a2, a3, a4);
}

extern void shmem_long_put_(long * a1, const long * a2, size_t a3, int a4)
{
   __wrap_shmem_long_put(a1, a2, a3, a4);
}

extern void shmem_long_put__(long * a1, const long * a2, size_t a3, int a4)
{
   __wrap_shmem_long_put(a1, a2, a3, a4);
}

extern void SHMEM_LONG_PUT_(long * a1, const long * a2, size_t a3, int a4)
{
   __wrap_shmem_long_put(a1, a2, a3, a4);
}

extern void SHMEM_LONG_PUT__(long * a1, const long * a2, size_t a3, int a4)
{
   __wrap_shmem_long_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_put
 **********************************************************/

extern void  __wrap_shmem_longlong_put(long long * a1, const long long * a2, size_t a3, int a4) ;
extern void  __real_shmem_longlong_put(long long * a1, const long long * a2, size_t a3, int a4)  {

  typedef void (*shmem_longlong_put_t)(long long * a1, const long long * a2, size_t a3, int a4);
  shmem_longlong_put_t shmem_longlong_put_handle = (shmem_longlong_put_t)get_function_handle("shmem_longlong_put");
  shmem_longlong_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_put(long long * a1, const long long * a2, size_t a3, int a4) {
   __wrap_shmem_longlong_put(a1, a2, a3, a4);
}

extern void shmem_longlong_put_(long long * a1, const long long * a2, size_t a3, int a4)
{
   __wrap_shmem_longlong_put(a1, a2, a3, a4);
}

extern void shmem_longlong_put__(long long * a1, const long long * a2, size_t a3, int a4)
{
   __wrap_shmem_longlong_put(a1, a2, a3, a4);
}

extern void SHMEM_LONGLONG_PUT_(long long * a1, const long long * a2, size_t a3, int a4)
{
   __wrap_shmem_longlong_put(a1, a2, a3, a4);
}

extern void SHMEM_LONGLONG_PUT__(long long * a1, const long long * a2, size_t a3, int a4)
{
   __wrap_shmem_longlong_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float_put
 **********************************************************/

extern void  __wrap_shmem_float_put(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __real_shmem_float_put(float * a1, const float * a2, size_t a3, int a4)  {

  typedef void (*shmem_float_put_t)(float * a1, const float * a2, size_t a3, int a4);
  shmem_float_put_t shmem_float_put_handle = (shmem_float_put_t)get_function_handle("shmem_float_put");
  shmem_float_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float_put(float * a1, const float * a2, size_t a3, int a4) {
   __wrap_shmem_float_put(a1, a2, a3, a4);
}

extern void shmem_float_put_(float * a1, const float * a2, size_t a3, int a4)
{
   __wrap_shmem_float_put(a1, a2, a3, a4);
}

extern void shmem_float_put__(float * a1, const float * a2, size_t a3, int a4)
{
   __wrap_shmem_float_put(a1, a2, a3, a4);
}

extern void SHMEM_FLOAT_PUT_(float * a1, const float * a2, size_t a3, int a4)
{
   __wrap_shmem_float_put(a1, a2, a3, a4);
}

extern void SHMEM_FLOAT_PUT__(float * a1, const float * a2, size_t a3, int a4)
{
   __wrap_shmem_float_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_double_put
 **********************************************************/

extern void  __wrap_shmem_double_put(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __real_shmem_double_put(double * a1, const double * a2, size_t a3, int a4)  {

  typedef void (*shmem_double_put_t)(double * a1, const double * a2, size_t a3, int a4);
  shmem_double_put_t shmem_double_put_handle = (shmem_double_put_t)get_function_handle("shmem_double_put");
  shmem_double_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_double_put(double * a1, const double * a2, size_t a3, int a4) {
   __wrap_shmem_double_put(a1, a2, a3, a4);
}

extern void shmem_double_put_(double * a1, const double * a2, size_t a3, int a4)
{
   __wrap_shmem_double_put(a1, a2, a3, a4);
}

extern void shmem_double_put__(double * a1, const double * a2, size_t a3, int a4)
{
   __wrap_shmem_double_put(a1, a2, a3, a4);
}

extern void SHMEM_DOUBLE_PUT_(double * a1, const double * a2, size_t a3, int a4)
{
   __wrap_shmem_double_put(a1, a2, a3, a4);
}

extern void SHMEM_DOUBLE_PUT__(double * a1, const double * a2, size_t a3, int a4)
{
   __wrap_shmem_double_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float128_put
 **********************************************************/

extern void  __wrap_shmem_float128_put(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __real_shmem_float128_put(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  typedef void (*shmem_float128_put_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4);
  shmem_float128_put_t shmem_float128_put_handle = (shmem_float128_put_t)get_function_handle("shmem_float128_put");
  shmem_float128_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float128_put(__float128 * a1, const __float128 * a2, size_t a3, int a4) {
   __wrap_shmem_float128_put(a1, a2, a3, a4);
}

extern void shmem_float128_put_(__float128 * a1, const __float128 * a2, size_t a3, int a4)
{
   __wrap_shmem_float128_put(a1, a2, a3, a4);
}

extern void shmem_float128_put__(__float128 * a1, const __float128 * a2, size_t a3, int a4)
{
   __wrap_shmem_float128_put(a1, a2, a3, a4);
}

extern void SHMEM_FLOAT128_PUT_(__float128 * a1, const __float128 * a2, size_t a3, int a4)
{
   __wrap_shmem_float128_put(a1, a2, a3, a4);
}

extern void SHMEM_FLOAT128_PUT__(__float128 * a1, const __float128 * a2, size_t a3, int a4)
{
   __wrap_shmem_float128_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_put_signal
 **********************************************************/

extern void  __wrap_shmem_short_put_signal(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_short_put_signal(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_short_put_signal_t)(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  shmem_short_put_signal_t shmem_short_put_signal_handle = (shmem_short_put_signal_t)get_function_handle("shmem_short_put_signal");
  shmem_short_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_short_put_signal(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_short_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_short_put_signal_(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_short_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_short_put_signal__(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_short_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_SHORT_PUT_SIGNAL_(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_short_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_SHORT_PUT_SIGNAL__(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_short_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_int_put_signal
 **********************************************************/

extern void  __wrap_shmem_int_put_signal(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_int_put_signal(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_int_put_signal_t)(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  shmem_int_put_signal_t shmem_int_put_signal_handle = (shmem_int_put_signal_t)get_function_handle("shmem_int_put_signal");
  shmem_int_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_int_put_signal(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_int_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_int_put_signal_(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_int_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_int_put_signal__(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_int_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_INT_PUT_SIGNAL_(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_int_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_INT_PUT_SIGNAL__(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_int_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_long_put_signal
 **********************************************************/

extern void  __wrap_shmem_long_put_signal(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_long_put_signal(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_long_put_signal_t)(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  shmem_long_put_signal_t shmem_long_put_signal_handle = (shmem_long_put_signal_t)get_function_handle("shmem_long_put_signal");
  shmem_long_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_long_put_signal(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_long_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_long_put_signal_(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_long_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_long_put_signal__(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_long_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONG_PUT_SIGNAL_(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_long_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONG_PUT_SIGNAL__(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_long_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_longlong_put_signal
 **********************************************************/

extern void  __wrap_shmem_longlong_put_signal(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_longlong_put_signal(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_longlong_put_signal_t)(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  shmem_longlong_put_signal_t shmem_longlong_put_signal_handle = (shmem_longlong_put_signal_t)get_function_handle("shmem_longlong_put_signal");
  shmem_longlong_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longlong_put_signal(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_longlong_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_longlong_put_signal_(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_longlong_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_longlong_put_signal__(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_longlong_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONGLONG_PUT_SIGNAL_(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_longlong_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONGLONG_PUT_SIGNAL__(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_longlong_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float_put_signal
 **********************************************************/

extern void  __wrap_shmem_float_put_signal(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_float_put_signal(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_float_put_signal_t)(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  shmem_float_put_signal_t shmem_float_put_signal_handle = (shmem_float_put_signal_t)get_function_handle("shmem_float_put_signal");
  shmem_float_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float_put_signal(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_float_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float_put_signal_(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_float_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float_put_signal__(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_float_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT_PUT_SIGNAL_(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_float_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT_PUT_SIGNAL__(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_float_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_double_put_signal
 **********************************************************/

extern void  __wrap_shmem_double_put_signal(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_double_put_signal(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_double_put_signal_t)(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  shmem_double_put_signal_t shmem_double_put_signal_handle = (shmem_double_put_signal_t)get_function_handle("shmem_double_put_signal");
  shmem_double_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_double_put_signal(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_double_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_double_put_signal_(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_double_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void shmem_double_put_signal__(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_double_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_DOUBLE_PUT_SIGNAL_(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_double_put_signal(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_DOUBLE_PUT_SIGNAL__(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)
{
   __wrap_shmem_double_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_short_get_nb
 **********************************************************/

extern void  __wrap_shmem_short_get_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_short_get_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_short_get_nb_t)(short * a1, const short * a2, size_t a3, int a4, void ** a5);
  shmem_short_get_nb_t shmem_short_get_nb_handle = (shmem_short_get_nb_t)get_function_handle("shmem_short_get_nb");
  shmem_short_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_short_get_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_short_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_short_get_nb_(short * a1, const short * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_short_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_short_get_nb__(short * a1, const short * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_short_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_SHORT_GET_NB_(short * a1, const short * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_short_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_SHORT_GET_NB__(short * a1, const short * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_short_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_int_get_nb
 **********************************************************/

extern void  __wrap_shmem_int_get_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_int_get_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_int_get_nb_t)(int * a1, const int * a2, size_t a3, int a4, void ** a5);
  shmem_int_get_nb_t shmem_int_get_nb_handle = (shmem_int_get_nb_t)get_function_handle("shmem_int_get_nb");
  shmem_int_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_int_get_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_int_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_int_get_nb_(int * a1, const int * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_int_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_int_get_nb__(int * a1, const int * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_int_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_INT_GET_NB_(int * a1, const int * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_int_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_INT_GET_NB__(int * a1, const int * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_int_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_long_get_nb
 **********************************************************/

extern void  __wrap_shmem_long_get_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_long_get_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_long_get_nb_t)(long * a1, const long * a2, size_t a3, int a4, void ** a5);
  shmem_long_get_nb_t shmem_long_get_nb_handle = (shmem_long_get_nb_t)get_function_handle("shmem_long_get_nb");
  shmem_long_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_long_get_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_long_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_long_get_nb_(long * a1, const long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_long_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_long_get_nb__(long * a1, const long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_long_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONG_GET_NB_(long * a1, const long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_long_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONG_GET_NB__(long * a1, const long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_long_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_longlong_get_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_longlong_get_nb_t)(long long * a1, const long long * a2, size_t a3, int a4, void ** a5);
  shmem_longlong_get_nb_t shmem_longlong_get_nb_handle = (shmem_longlong_get_nb_t)get_function_handle("shmem_longlong_get_nb");
  shmem_longlong_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_longlong_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_longlong_get_nb_(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_longlong_get_nb__(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONGLONG_GET_NB_(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONGLONG_GET_NB__(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float_get_nb
 **********************************************************/

extern void  __wrap_shmem_float_get_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_float_get_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_float_get_nb_t)(float * a1, const float * a2, size_t a3, int a4, void ** a5);
  shmem_float_get_nb_t shmem_float_get_nb_handle = (shmem_float_get_nb_t)get_function_handle("shmem_float_get_nb");
  shmem_float_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float_get_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_float_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float_get_nb_(float * a1, const float * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float_get_nb__(float * a1, const float * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT_GET_NB_(float * a1, const float * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT_GET_NB__(float * a1, const float * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_double_get_nb
 **********************************************************/

extern void  __wrap_shmem_double_get_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_double_get_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_double_get_nb_t)(double * a1, const double * a2, size_t a3, int a4, void ** a5);
  shmem_double_get_nb_t shmem_double_get_nb_handle = (shmem_double_get_nb_t)get_function_handle("shmem_double_get_nb");
  shmem_double_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_double_get_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_double_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_double_get_nb_(double * a1, const double * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_double_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_double_get_nb__(double * a1, const double * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_double_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_DOUBLE_GET_NB_(double * a1, const double * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_double_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_DOUBLE_GET_NB__(double * a1, const double * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_double_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float128_get_nb
 **********************************************************/

extern void  __wrap_shmem_float128_get_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_float128_get_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_float128_get_nb_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5);
  shmem_float128_get_nb_t shmem_float128_get_nb_handle = (shmem_float128_get_nb_t)get_function_handle("shmem_float128_get_nb");
  shmem_float128_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float128_get_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_float128_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float128_get_nb_(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float128_get_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float128_get_nb__(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float128_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT128_GET_NB_(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float128_get_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT128_GET_NB__(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float128_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_short_put_nb
 **********************************************************/

extern void  __wrap_shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_short_put_nb_t)(short * a1, const short * a2, size_t a3, int a4, void ** a5);
  shmem_short_put_nb_t shmem_short_put_nb_handle = (shmem_short_put_nb_t)get_function_handle("shmem_short_put_nb");
  shmem_short_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_short_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_short_put_nb_(short * a1, const short * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_short_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_short_put_nb__(short * a1, const short * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_short_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_SHORT_PUT_NB_(short * a1, const short * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_short_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_SHORT_PUT_NB__(short * a1, const short * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_short_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_int_put_nb
 **********************************************************/

extern void  __wrap_shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_int_put_nb_t)(int * a1, const int * a2, size_t a3, int a4, void ** a5);
  shmem_int_put_nb_t shmem_int_put_nb_handle = (shmem_int_put_nb_t)get_function_handle("shmem_int_put_nb");
  shmem_int_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_int_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_int_put_nb_(int * a1, const int * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_int_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_int_put_nb__(int * a1, const int * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_int_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_INT_PUT_NB_(int * a1, const int * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_int_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_INT_PUT_NB__(int * a1, const int * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_int_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_long_put_nb
 **********************************************************/

extern void  __wrap_shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_long_put_nb_t)(long * a1, const long * a2, size_t a3, int a4, void ** a5);
  shmem_long_put_nb_t shmem_long_put_nb_handle = (shmem_long_put_nb_t)get_function_handle("shmem_long_put_nb");
  shmem_long_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_long_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_long_put_nb_(long * a1, const long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_long_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_long_put_nb__(long * a1, const long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_long_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONG_PUT_NB_(long * a1, const long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_long_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONG_PUT_NB__(long * a1, const long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_long_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_longlong_put_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_longlong_put_nb_t)(long long * a1, const long long * a2, size_t a3, int a4, void ** a5);
  shmem_longlong_put_nb_t shmem_longlong_put_nb_handle = (shmem_longlong_put_nb_t)get_function_handle("shmem_longlong_put_nb");
  shmem_longlong_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_longlong_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_longlong_put_nb_(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_longlong_put_nb__(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONGLONG_PUT_NB_(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONGLONG_PUT_NB__(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float_put_nb
 **********************************************************/

extern void  __wrap_shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_float_put_nb_t)(float * a1, const float * a2, size_t a3, int a4, void ** a5);
  shmem_float_put_nb_t shmem_float_put_nb_handle = (shmem_float_put_nb_t)get_function_handle("shmem_float_put_nb");
  shmem_float_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_float_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float_put_nb_(float * a1, const float * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float_put_nb__(float * a1, const float * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT_PUT_NB_(float * a1, const float * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT_PUT_NB__(float * a1, const float * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_double_put_nb
 **********************************************************/

extern void  __wrap_shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_double_put_nb_t)(double * a1, const double * a2, size_t a3, int a4, void ** a5);
  shmem_double_put_nb_t shmem_double_put_nb_handle = (shmem_double_put_nb_t)get_function_handle("shmem_double_put_nb");
  shmem_double_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_double_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_double_put_nb_(double * a1, const double * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_double_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_double_put_nb__(double * a1, const double * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_double_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_DOUBLE_PUT_NB_(double * a1, const double * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_double_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_DOUBLE_PUT_NB__(double * a1, const double * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_double_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float128_put_nb
 **********************************************************/

extern void  __wrap_shmem_float128_put_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_float128_put_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_float128_put_nb_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5);
  shmem_float128_put_nb_t shmem_float128_put_nb_handle = (shmem_float128_put_nb_t)get_function_handle("shmem_float128_put_nb");
  shmem_float128_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float128_put_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_float128_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float128_put_nb_(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float128_put_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float128_put_nb__(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float128_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT128_PUT_NB_(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float128_put_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT128_PUT_NB__(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)
{
   __wrap_shmem_float128_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_short_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_short_put_signal_nb(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_short_put_signal_nb(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_short_put_signal_nb_t)(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  shmem_short_put_signal_nb_t shmem_short_put_signal_nb_handle = (shmem_short_put_signal_nb_t)get_function_handle("shmem_short_put_signal_nb");
  shmem_short_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_short_put_signal_nb(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_short_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_short_put_signal_nb_(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_short_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_short_put_signal_nb__(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_short_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_SHORT_PUT_SIGNAL_NB_(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_short_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_SHORT_PUT_SIGNAL_NB__(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_short_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_int_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_int_put_signal_nb(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_int_put_signal_nb(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_int_put_signal_nb_t)(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  shmem_int_put_signal_nb_t shmem_int_put_signal_nb_handle = (shmem_int_put_signal_nb_t)get_function_handle("shmem_int_put_signal_nb");
  shmem_int_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_int_put_signal_nb(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_int_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_int_put_signal_nb_(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_int_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_int_put_signal_nb__(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_int_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_INT_PUT_SIGNAL_NB_(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_int_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_INT_PUT_SIGNAL_NB__(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_int_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_long_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_long_put_signal_nb(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_long_put_signal_nb(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_long_put_signal_nb_t)(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  shmem_long_put_signal_nb_t shmem_long_put_signal_nb_handle = (shmem_long_put_signal_nb_t)get_function_handle("shmem_long_put_signal_nb");
  shmem_long_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_long_put_signal_nb(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_long_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_long_put_signal_nb_(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_long_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_long_put_signal_nb__(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_long_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_LONG_PUT_SIGNAL_NB_(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_long_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_LONG_PUT_SIGNAL_NB__(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_long_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_longlong_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_put_signal_nb(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_longlong_put_signal_nb(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_longlong_put_signal_nb_t)(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  shmem_longlong_put_signal_nb_t shmem_longlong_put_signal_nb_handle = (shmem_longlong_put_signal_nb_t)get_function_handle("shmem_longlong_put_signal_nb");
  shmem_longlong_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_longlong_put_signal_nb(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_longlong_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_longlong_put_signal_nb_(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_longlong_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_longlong_put_signal_nb__(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_longlong_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_LONGLONG_PUT_SIGNAL_NB_(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_longlong_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_LONGLONG_PUT_SIGNAL_NB__(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_longlong_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_float_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_float_put_signal_nb(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_float_put_signal_nb(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_float_put_signal_nb_t)(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  shmem_float_put_signal_nb_t shmem_float_put_signal_nb_handle = (shmem_float_put_signal_nb_t)get_function_handle("shmem_float_put_signal_nb");
  shmem_float_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_float_put_signal_nb(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_float_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_float_put_signal_nb_(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_float_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_float_put_signal_nb__(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_float_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_FLOAT_PUT_SIGNAL_NB_(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_float_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_FLOAT_PUT_SIGNAL_NB__(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_float_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_double_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_double_put_signal_nb(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_double_put_signal_nb(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_double_put_signal_nb_t)(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  shmem_double_put_signal_nb_t shmem_double_put_signal_nb_handle = (shmem_double_put_signal_nb_t)get_function_handle("shmem_double_put_signal_nb");
  shmem_double_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_double_put_signal_nb(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_double_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_double_put_signal_nb_(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_double_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_double_put_signal_nb__(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_double_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_DOUBLE_PUT_SIGNAL_NB_(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_double_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_DOUBLE_PUT_SIGNAL_NB__(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)
{
   __wrap_shmem_double_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_short_iget
 **********************************************************/

extern void  __wrap_shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_short_iget_t)(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_short_iget_t shmem_short_iget_handle = (shmem_short_iget_t)get_function_handle("shmem_short_iget");
  shmem_short_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_short_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_short_iget_(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_short_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_short_iget__(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_short_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_SHORT_IGET_(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_short_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_SHORT_IGET__(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_short_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_int_iget
 **********************************************************/

extern void  __wrap_shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_int_iget_t)(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_int_iget_t shmem_int_iget_handle = (shmem_int_iget_t)get_function_handle("shmem_int_iget");
  shmem_int_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_int_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_int_iget_(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_int_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_int_iget__(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_int_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_INT_IGET_(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_int_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_INT_IGET__(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_int_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_long_iget
 **********************************************************/

extern void  __wrap_shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_long_iget_t)(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_long_iget_t shmem_long_iget_handle = (shmem_long_iget_t)get_function_handle("shmem_long_iget");
  shmem_long_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_long_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_long_iget_(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_long_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_long_iget__(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_long_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONG_IGET_(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_long_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONG_IGET__(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_long_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_longlong_iget
 **********************************************************/

extern void  __wrap_shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_longlong_iget_t)(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_longlong_iget_t shmem_longlong_iget_handle = (shmem_longlong_iget_t)get_function_handle("shmem_longlong_iget");
  shmem_longlong_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_longlong_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_longlong_iget_(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_longlong_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_longlong_iget__(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_longlong_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONGLONG_IGET_(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_longlong_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONGLONG_IGET__(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_longlong_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float_iget
 **********************************************************/

extern void  __wrap_shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_float_iget_t)(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_float_iget_t shmem_float_iget_handle = (shmem_float_iget_t)get_function_handle("shmem_float_iget");
  shmem_float_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_float_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float_iget_(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float_iget__(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT_IGET_(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT_IGET__(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_double_iget
 **********************************************************/

extern void  __wrap_shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_double_iget_t)(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_double_iget_t shmem_double_iget_handle = (shmem_double_iget_t)get_function_handle("shmem_double_iget");
  shmem_double_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_double_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_double_iget_(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_double_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_double_iget__(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_double_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_DOUBLE_IGET_(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_double_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_DOUBLE_IGET__(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_double_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float128_iget
 **********************************************************/

extern void  __wrap_shmem_float128_iget(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_float128_iget(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_float128_iget_t)(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_float128_iget_t shmem_float128_iget_handle = (shmem_float128_iget_t)get_function_handle("shmem_float128_iget");
  shmem_float128_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float128_iget(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_float128_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float128_iget_(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float128_iget(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float128_iget__(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float128_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT128_IGET_(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float128_iget(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT128_IGET__(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float128_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_short_iput
 **********************************************************/

extern void  __wrap_shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_short_iput_t)(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_short_iput_t shmem_short_iput_handle = (shmem_short_iput_t)get_function_handle("shmem_short_iput");
  shmem_short_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_short_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_short_iput_(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_short_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_short_iput__(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_short_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_SHORT_IPUT_(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_short_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_SHORT_IPUT__(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_short_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_int_iput
 **********************************************************/

extern void  __wrap_shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_int_iput_t)(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_int_iput_t shmem_int_iput_handle = (shmem_int_iput_t)get_function_handle("shmem_int_iput");
  shmem_int_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_int_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_int_iput_(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_int_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_int_iput__(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_int_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_INT_IPUT_(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_int_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_INT_IPUT__(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_int_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_long_iput
 **********************************************************/

extern void  __wrap_shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_long_iput_t)(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_long_iput_t shmem_long_iput_handle = (shmem_long_iput_t)get_function_handle("shmem_long_iput");
  shmem_long_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_long_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_long_iput_(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_long_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_long_iput__(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_long_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONG_IPUT_(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_long_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONG_IPUT__(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_long_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_longlong_iput
 **********************************************************/

extern void  __wrap_shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_longlong_iput_t)(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_longlong_iput_t shmem_longlong_iput_handle = (shmem_longlong_iput_t)get_function_handle("shmem_longlong_iput");
  shmem_longlong_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_longlong_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_longlong_iput_(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_longlong_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_longlong_iput__(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_longlong_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONGLONG_IPUT_(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_longlong_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONGLONG_IPUT__(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_longlong_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float_iput
 **********************************************************/

extern void  __wrap_shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_float_iput_t)(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_float_iput_t shmem_float_iput_handle = (shmem_float_iput_t)get_function_handle("shmem_float_iput");
  shmem_float_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_float_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float_iput_(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float_iput__(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT_IPUT_(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT_IPUT__(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_double_iput
 **********************************************************/

extern void  __wrap_shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_double_iput_t)(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_double_iput_t shmem_double_iput_handle = (shmem_double_iput_t)get_function_handle("shmem_double_iput");
  shmem_double_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_double_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_double_iput_(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_double_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_double_iput__(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_double_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_DOUBLE_IPUT_(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_double_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_DOUBLE_IPUT__(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_double_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float128_iput
 **********************************************************/

extern void  __wrap_shmem_float128_iput(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_float128_iput(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_float128_iput_t)(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_float128_iput_t shmem_float128_iput_handle = (shmem_float128_iput_t)get_function_handle("shmem_float128_iput");
  shmem_float128_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float128_iput(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_float128_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float128_iput_(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float128_iput(a1, a2, a3, a4, a5, a6);
}

extern void shmem_float128_iput__(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float128_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT128_IPUT_(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float128_iput(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_FLOAT128_IPUT__(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)
{
   __wrap_shmem_float128_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_char_g
 **********************************************************/

extern char  __wrap_shmem_char_g(const char * a1, int a2) ;
extern char  __real_shmem_char_g(const char * a1, int a2)  {

  char retval;
  typedef char (*shmem_char_g_t)(const char * a1, int a2);
  shmem_char_g_t shmem_char_g_handle = (shmem_char_g_t)get_function_handle("shmem_char_g");
  retval  =  shmem_char_g_handle ( a1,  a2);
  return retval;

}

extern char  shmem_char_g(const char * a1, int a2) {
   __wrap_shmem_char_g(a1, a2);
}

extern char shmem_char_g_(const char * a1, int a2)
{
   __wrap_shmem_char_g(a1, a2);
}

extern char shmem_char_g__(const char * a1, int a2)
{
   __wrap_shmem_char_g(a1, a2);
}

extern char SHMEM_CHAR_G_(const char * a1, int a2)
{
   __wrap_shmem_char_g(a1, a2);
}

extern char SHMEM_CHAR_G__(const char * a1, int a2)
{
   __wrap_shmem_char_g(a1, a2);
}


/**********************************************************
   shmem_short_g
 **********************************************************/

extern short  __wrap_shmem_short_g(const short * a1, int a2) ;
extern short  __real_shmem_short_g(const short * a1, int a2)  {

  short retval;
  typedef short (*shmem_short_g_t)(const short * a1, int a2);
  shmem_short_g_t shmem_short_g_handle = (shmem_short_g_t)get_function_handle("shmem_short_g");
  retval  =  shmem_short_g_handle ( a1,  a2);
  return retval;

}

extern short  shmem_short_g(const short * a1, int a2) {
   __wrap_shmem_short_g(a1, a2);
}

extern short shmem_short_g_(const short * a1, int a2)
{
   __wrap_shmem_short_g(a1, a2);
}

extern short shmem_short_g__(const short * a1, int a2)
{
   __wrap_shmem_short_g(a1, a2);
}

extern short SHMEM_SHORT_G_(const short * a1, int a2)
{
   __wrap_shmem_short_g(a1, a2);
}

extern short SHMEM_SHORT_G__(const short * a1, int a2)
{
   __wrap_shmem_short_g(a1, a2);
}


/**********************************************************
   shmem_int_g
 **********************************************************/

extern int  __wrap_shmem_int_g(const int * a1, int a2) ;
extern int  __real_shmem_int_g(const int * a1, int a2)  {

  int retval;
  typedef int (*shmem_int_g_t)(const int * a1, int a2);
  shmem_int_g_t shmem_int_g_handle = (shmem_int_g_t)get_function_handle("shmem_int_g");
  retval  =  shmem_int_g_handle ( a1,  a2);
  return retval;

}

extern int  shmem_int_g(const int * a1, int a2) {
   __wrap_shmem_int_g(a1, a2);
}

extern int shmem_int_g_(const int * a1, int a2)
{
   __wrap_shmem_int_g(a1, a2);
}

extern int shmem_int_g__(const int * a1, int a2)
{
   __wrap_shmem_int_g(a1, a2);
}

extern int SHMEM_INT_G_(const int * a1, int a2)
{
   __wrap_shmem_int_g(a1, a2);
}

extern int SHMEM_INT_G__(const int * a1, int a2)
{
   __wrap_shmem_int_g(a1, a2);
}


/**********************************************************
   shmem_long_g
 **********************************************************/

extern long  __wrap_shmem_long_g(const long * a1, int a2) ;
extern long  __real_shmem_long_g(const long * a1, int a2)  {

  long retval;
  typedef long (*shmem_long_g_t)(const long * a1, int a2);
  shmem_long_g_t shmem_long_g_handle = (shmem_long_g_t)get_function_handle("shmem_long_g");
  retval  =  shmem_long_g_handle ( a1,  a2);
  return retval;

}

extern long  shmem_long_g(const long * a1, int a2) {
   __wrap_shmem_long_g(a1, a2);
}

extern long shmem_long_g_(const long * a1, int a2)
{
   __wrap_shmem_long_g(a1, a2);
}

extern long shmem_long_g__(const long * a1, int a2)
{
   __wrap_shmem_long_g(a1, a2);
}

extern long SHMEM_LONG_G_(const long * a1, int a2)
{
   __wrap_shmem_long_g(a1, a2);
}

extern long SHMEM_LONG_G__(const long * a1, int a2)
{
   __wrap_shmem_long_g(a1, a2);
}


/**********************************************************
   shmem_longlong_g
 **********************************************************/

extern long long  __wrap_shmem_longlong_g(const long long * a1, int a2) ;
extern long long  __real_shmem_longlong_g(const long long * a1, int a2)  {

  long long retval;
  typedef long long (*shmem_longlong_g_t)(const long long * a1, int a2);
  shmem_longlong_g_t shmem_longlong_g_handle = (shmem_longlong_g_t)get_function_handle("shmem_longlong_g");
  retval  =  shmem_longlong_g_handle ( a1,  a2);
  return retval;

}

extern long long  shmem_longlong_g(const long long * a1, int a2) {
   __wrap_shmem_longlong_g(a1, a2);
}

extern long long shmem_longlong_g_(const long long * a1, int a2)
{
   __wrap_shmem_longlong_g(a1, a2);
}

extern long long shmem_longlong_g__(const long long * a1, int a2)
{
   __wrap_shmem_longlong_g(a1, a2);
}

extern long long SHMEM_LONGLONG_G_(const long long * a1, int a2)
{
   __wrap_shmem_longlong_g(a1, a2);
}

extern long long SHMEM_LONGLONG_G__(const long long * a1, int a2)
{
   __wrap_shmem_longlong_g(a1, a2);
}


/**********************************************************
   shmem_float_g
 **********************************************************/

extern float  __wrap_shmem_float_g(const float * a1, int a2) ;
extern float  __real_shmem_float_g(const float * a1, int a2)  {

  float retval;
  typedef float (*shmem_float_g_t)(const float * a1, int a2);
  shmem_float_g_t shmem_float_g_handle = (shmem_float_g_t)get_function_handle("shmem_float_g");
  retval  =  shmem_float_g_handle ( a1,  a2);
  return retval;

}

extern float  shmem_float_g(const float * a1, int a2) {
   __wrap_shmem_float_g(a1, a2);
}

extern float shmem_float_g_(const float * a1, int a2)
{
   __wrap_shmem_float_g(a1, a2);
}

extern float shmem_float_g__(const float * a1, int a2)
{
   __wrap_shmem_float_g(a1, a2);
}

extern float SHMEM_FLOAT_G_(const float * a1, int a2)
{
   __wrap_shmem_float_g(a1, a2);
}

extern float SHMEM_FLOAT_G__(const float * a1, int a2)
{
   __wrap_shmem_float_g(a1, a2);
}


/**********************************************************
   shmem_double_g
 **********************************************************/

extern double  __wrap_shmem_double_g(const double * a1, int a2) ;
extern double  __real_shmem_double_g(const double * a1, int a2)  {

  double retval;
  typedef double (*shmem_double_g_t)(const double * a1, int a2);
  shmem_double_g_t shmem_double_g_handle = (shmem_double_g_t)get_function_handle("shmem_double_g");
  retval  =  shmem_double_g_handle ( a1,  a2);
  return retval;

}

extern double  shmem_double_g(const double * a1, int a2) {
   __wrap_shmem_double_g(a1, a2);
}

extern double shmem_double_g_(const double * a1, int a2)
{
   __wrap_shmem_double_g(a1, a2);
}

extern double shmem_double_g__(const double * a1, int a2)
{
   __wrap_shmem_double_g(a1, a2);
}

extern double SHMEM_DOUBLE_G_(const double * a1, int a2)
{
   __wrap_shmem_double_g(a1, a2);
}

extern double SHMEM_DOUBLE_G__(const double * a1, int a2)
{
   __wrap_shmem_double_g(a1, a2);
}


/**********************************************************
   shmem_ld80_g
 **********************************************************/

extern long double  __wrap_shmem_ld80_g(const long double * a1, int a2) ;
extern long double  __real_shmem_ld80_g(const long double * a1, int a2)  {

  long double retval;
  typedef long double (*shmem_ld80_g_t)(const long double * a1, int a2);
  shmem_ld80_g_t shmem_ld80_g_handle = (shmem_ld80_g_t)get_function_handle("shmem_ld80_g");
  retval  =  shmem_ld80_g_handle ( a1,  a2);
  return retval;

}

extern long double  shmem_ld80_g(const long double * a1, int a2) {
   __wrap_shmem_ld80_g(a1, a2);
}

extern long double shmem_ld80_g_(const long double * a1, int a2)
{
   __wrap_shmem_ld80_g(a1, a2);
}

extern long double shmem_ld80_g__(const long double * a1, int a2)
{
   __wrap_shmem_ld80_g(a1, a2);
}

extern long double SHMEM_LD80_G_(const long double * a1, int a2)
{
   __wrap_shmem_ld80_g(a1, a2);
}

extern long double SHMEM_LD80_G__(const long double * a1, int a2)
{
   __wrap_shmem_ld80_g(a1, a2);
}


/**********************************************************
   shmem_float128_g
 **********************************************************/

extern __float128  __wrap_shmem_float128_g(const __float128 * a1, int a2) ;
extern __float128  __real_shmem_float128_g(const __float128 * a1, int a2)  {

  __float128 retval;
  typedef __float128 (*shmem_float128_g_t)(const __float128 * a1, int a2);
  shmem_float128_g_t shmem_float128_g_handle = (shmem_float128_g_t)get_function_handle("shmem_float128_g");
  retval  =  shmem_float128_g_handle ( a1,  a2);
  return retval;

}

extern __float128  shmem_float128_g(const __float128 * a1, int a2) {
   __wrap_shmem_float128_g(a1, a2);
}

extern __float128 shmem_float128_g_(const __float128 * a1, int a2)
{
   __wrap_shmem_float128_g(a1, a2);
}

extern __float128 shmem_float128_g__(const __float128 * a1, int a2)
{
   __wrap_shmem_float128_g(a1, a2);
}

extern __float128 SHMEM_FLOAT128_G_(const __float128 * a1, int a2)
{
   __wrap_shmem_float128_g(a1, a2);
}

extern __float128 SHMEM_FLOAT128_G__(const __float128 * a1, int a2)
{
   __wrap_shmem_float128_g(a1, a2);
}


/**********************************************************
   shmem_char_p
 **********************************************************/

extern void  __wrap_shmem_char_p(char * a1, char a2, int a3) ;
extern void  __real_shmem_char_p(char * a1, char a2, int a3)  {

  typedef void (*shmem_char_p_t)(char * a1, char a2, int a3);
  shmem_char_p_t shmem_char_p_handle = (shmem_char_p_t)get_function_handle("shmem_char_p");
  shmem_char_p_handle ( a1,  a2,  a3);

}

extern void  shmem_char_p(char * a1, char a2, int a3) {
   __wrap_shmem_char_p(a1, a2, a3);
}

extern void shmem_char_p_(char * a1, char a2, int a3)
{
   __wrap_shmem_char_p(a1, a2, a3);
}

extern void shmem_char_p__(char * a1, char a2, int a3)
{
   __wrap_shmem_char_p(a1, a2, a3);
}

extern void SHMEM_CHAR_P_(char * a1, char a2, int a3)
{
   __wrap_shmem_char_p(a1, a2, a3);
}

extern void SHMEM_CHAR_P__(char * a1, char a2, int a3)
{
   __wrap_shmem_char_p(a1, a2, a3);
}


/**********************************************************
   shmem_short_p
 **********************************************************/

extern void  __wrap_shmem_short_p(short * a1, short a2, int a3) ;
extern void  __real_shmem_short_p(short * a1, short a2, int a3)  {

  typedef void (*shmem_short_p_t)(short * a1, short a2, int a3);
  shmem_short_p_t shmem_short_p_handle = (shmem_short_p_t)get_function_handle("shmem_short_p");
  shmem_short_p_handle ( a1,  a2,  a3);

}

extern void  shmem_short_p(short * a1, short a2, int a3) {
   __wrap_shmem_short_p(a1, a2, a3);
}

extern void shmem_short_p_(short * a1, short a2, int a3)
{
   __wrap_shmem_short_p(a1, a2, a3);
}

extern void shmem_short_p__(short * a1, short a2, int a3)
{
   __wrap_shmem_short_p(a1, a2, a3);
}

extern void SHMEM_SHORT_P_(short * a1, short a2, int a3)
{
   __wrap_shmem_short_p(a1, a2, a3);
}

extern void SHMEM_SHORT_P__(short * a1, short a2, int a3)
{
   __wrap_shmem_short_p(a1, a2, a3);
}


/**********************************************************
   shmem_int_p
 **********************************************************/

extern void  __wrap_shmem_int_p(int * a1, int a2, int a3) ;
extern void  __real_shmem_int_p(int * a1, int a2, int a3)  {

  typedef void (*shmem_int_p_t)(int * a1, int a2, int a3);
  shmem_int_p_t shmem_int_p_handle = (shmem_int_p_t)get_function_handle("shmem_int_p");
  shmem_int_p_handle ( a1,  a2,  a3);

}

extern void  shmem_int_p(int * a1, int a2, int a3) {
   __wrap_shmem_int_p(a1, a2, a3);
}

extern void shmem_int_p_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_p(a1, a2, a3);
}

extern void shmem_int_p__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_p(a1, a2, a3);
}

extern void SHMEM_INT_P_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_p(a1, a2, a3);
}

extern void SHMEM_INT_P__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_p(a1, a2, a3);
}


/**********************************************************
   shmem_long_p
 **********************************************************/

extern void  __wrap_shmem_long_p(long * a1, long a2, int a3) ;
extern void  __real_shmem_long_p(long * a1, long a2, int a3)  {

  typedef void (*shmem_long_p_t)(long * a1, long a2, int a3);
  shmem_long_p_t shmem_long_p_handle = (shmem_long_p_t)get_function_handle("shmem_long_p");
  shmem_long_p_handle ( a1,  a2,  a3);

}

extern void  shmem_long_p(long * a1, long a2, int a3) {
   __wrap_shmem_long_p(a1, a2, a3);
}

extern void shmem_long_p_(long * a1, long a2, int a3)
{
   __wrap_shmem_long_p(a1, a2, a3);
}

extern void shmem_long_p__(long * a1, long a2, int a3)
{
   __wrap_shmem_long_p(a1, a2, a3);
}

extern void SHMEM_LONG_P_(long * a1, long a2, int a3)
{
   __wrap_shmem_long_p(a1, a2, a3);
}

extern void SHMEM_LONG_P__(long * a1, long a2, int a3)
{
   __wrap_shmem_long_p(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_p
 **********************************************************/

extern void  __wrap_shmem_longlong_p(long long * a1, long long a2, int a3) ;
extern void  __real_shmem_longlong_p(long long * a1, long long a2, int a3)  {

  typedef void (*shmem_longlong_p_t)(long long * a1, long long a2, int a3);
  shmem_longlong_p_t shmem_longlong_p_handle = (shmem_longlong_p_t)get_function_handle("shmem_longlong_p");
  shmem_longlong_p_handle ( a1,  a2,  a3);

}

extern void  shmem_longlong_p(long long * a1, long long a2, int a3) {
   __wrap_shmem_longlong_p(a1, a2, a3);
}

extern void shmem_longlong_p_(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_p(a1, a2, a3);
}

extern void shmem_longlong_p__(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_p(a1, a2, a3);
}

extern void SHMEM_LONGLONG_P_(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_p(a1, a2, a3);
}

extern void SHMEM_LONGLONG_P__(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_p(a1, a2, a3);
}


/**********************************************************
   shmem_float_p
 **********************************************************/

extern void  __wrap_shmem_float_p(float * a1, float a2, int a3) ;
extern void  __real_shmem_float_p(float * a1, float a2, int a3)  {

  typedef void (*shmem_float_p_t)(float * a1, float a2, int a3);
  shmem_float_p_t shmem_float_p_handle = (shmem_float_p_t)get_function_handle("shmem_float_p");
  shmem_float_p_handle ( a1,  a2,  a3);

}

extern void  shmem_float_p(float * a1, float a2, int a3) {
   __wrap_shmem_float_p(a1, a2, a3);
}

extern void shmem_float_p_(float * a1, float a2, int a3)
{
   __wrap_shmem_float_p(a1, a2, a3);
}

extern void shmem_float_p__(float * a1, float a2, int a3)
{
   __wrap_shmem_float_p(a1, a2, a3);
}

extern void SHMEM_FLOAT_P_(float * a1, float a2, int a3)
{
   __wrap_shmem_float_p(a1, a2, a3);
}

extern void SHMEM_FLOAT_P__(float * a1, float a2, int a3)
{
   __wrap_shmem_float_p(a1, a2, a3);
}


/**********************************************************
   shmem_double_p
 **********************************************************/

extern void  __wrap_shmem_double_p(double * a1, double a2, int a3) ;
extern void  __real_shmem_double_p(double * a1, double a2, int a3)  {

  typedef void (*shmem_double_p_t)(double * a1, double a2, int a3);
  shmem_double_p_t shmem_double_p_handle = (shmem_double_p_t)get_function_handle("shmem_double_p");
  shmem_double_p_handle ( a1,  a2,  a3);

}

extern void  shmem_double_p(double * a1, double a2, int a3) {
   __wrap_shmem_double_p(a1, a2, a3);
}

extern void shmem_double_p_(double * a1, double a2, int a3)
{
   __wrap_shmem_double_p(a1, a2, a3);
}

extern void shmem_double_p__(double * a1, double a2, int a3)
{
   __wrap_shmem_double_p(a1, a2, a3);
}

extern void SHMEM_DOUBLE_P_(double * a1, double a2, int a3)
{
   __wrap_shmem_double_p(a1, a2, a3);
}

extern void SHMEM_DOUBLE_P__(double * a1, double a2, int a3)
{
   __wrap_shmem_double_p(a1, a2, a3);
}


/**********************************************************
   shmem_ld80_p
 **********************************************************/

extern void  __wrap_shmem_ld80_p(long double * a1, long double a2, int a3) ;
extern void  __real_shmem_ld80_p(long double * a1, long double a2, int a3)  {

  typedef void (*shmem_ld80_p_t)(long double * a1, long double a2, int a3);
  shmem_ld80_p_t shmem_ld80_p_handle = (shmem_ld80_p_t)get_function_handle("shmem_ld80_p");
  shmem_ld80_p_handle ( a1,  a2,  a3);

}

extern void  shmem_ld80_p(long double * a1, long double a2, int a3) {
   __wrap_shmem_ld80_p(a1, a2, a3);
}

extern void shmem_ld80_p_(long double * a1, long double a2, int a3)
{
   __wrap_shmem_ld80_p(a1, a2, a3);
}

extern void shmem_ld80_p__(long double * a1, long double a2, int a3)
{
   __wrap_shmem_ld80_p(a1, a2, a3);
}

extern void SHMEM_LD80_P_(long double * a1, long double a2, int a3)
{
   __wrap_shmem_ld80_p(a1, a2, a3);
}

extern void SHMEM_LD80_P__(long double * a1, long double a2, int a3)
{
   __wrap_shmem_ld80_p(a1, a2, a3);
}


/**********************************************************
   shmem_float128_p
 **********************************************************/

extern void  __wrap_shmem_float128_p(__float128 * a1, __float128 a2, int a3) ;
extern void  __real_shmem_float128_p(__float128 * a1, __float128 a2, int a3)  {

  typedef void (*shmem_float128_p_t)(__float128 * a1, __float128 a2, int a3);
  shmem_float128_p_t shmem_float128_p_handle = (shmem_float128_p_t)get_function_handle("shmem_float128_p");
  shmem_float128_p_handle ( a1,  a2,  a3);

}

extern void  shmem_float128_p(__float128 * a1, __float128 a2, int a3) {
   __wrap_shmem_float128_p(a1, a2, a3);
}

extern void shmem_float128_p_(__float128 * a1, __float128 a2, int a3)
{
   __wrap_shmem_float128_p(a1, a2, a3);
}

extern void shmem_float128_p__(__float128 * a1, __float128 a2, int a3)
{
   __wrap_shmem_float128_p(a1, a2, a3);
}

extern void SHMEM_FLOAT128_P_(__float128 * a1, __float128 a2, int a3)
{
   __wrap_shmem_float128_p(a1, a2, a3);
}

extern void SHMEM_FLOAT128_P__(__float128 * a1, __float128 a2, int a3)
{
   __wrap_shmem_float128_p(a1, a2, a3);
}


/**********************************************************
   shmem_short_swap
 **********************************************************/

extern short  __wrap_shmem_short_swap(short * a1, short a2, int a3) ;
extern short  __real_shmem_short_swap(short * a1, short a2, int a3)  {

  short retval;
  typedef short (*shmem_short_swap_t)(short * a1, short a2, int a3);
  shmem_short_swap_t shmem_short_swap_handle = (shmem_short_swap_t)get_function_handle("shmem_short_swap");
  retval  =  shmem_short_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern short  shmem_short_swap(short * a1, short a2, int a3) {
   __wrap_shmem_short_swap(a1, a2, a3);
}

extern short shmem_short_swap_(short * a1, short a2, int a3)
{
   __wrap_shmem_short_swap(a1, a2, a3);
}

extern short shmem_short_swap__(short * a1, short a2, int a3)
{
   __wrap_shmem_short_swap(a1, a2, a3);
}

extern short SHMEM_SHORT_SWAP_(short * a1, short a2, int a3)
{
   __wrap_shmem_short_swap(a1, a2, a3);
}

extern short SHMEM_SHORT_SWAP__(short * a1, short a2, int a3)
{
   __wrap_shmem_short_swap(a1, a2, a3);
}


/**********************************************************
   shmem_int_swap
 **********************************************************/

extern int  __wrap_shmem_int_swap(int * a1, int a2, int a3) ;
extern int  __real_shmem_int_swap(int * a1, int a2, int a3)  {

  int retval;
  typedef int (*shmem_int_swap_t)(int * a1, int a2, int a3);
  shmem_int_swap_t shmem_int_swap_handle = (shmem_int_swap_t)get_function_handle("shmem_int_swap");
  retval  =  shmem_int_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern int  shmem_int_swap(int * a1, int a2, int a3) {
   __wrap_shmem_int_swap(a1, a2, a3);
}

extern int shmem_int_swap_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_swap(a1, a2, a3);
}

extern int shmem_int_swap__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_swap(a1, a2, a3);
}

extern int SHMEM_INT_SWAP_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_swap(a1, a2, a3);
}

extern int SHMEM_INT_SWAP__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_swap(a1, a2, a3);
}


/**********************************************************
   shmem_long_swap
 **********************************************************/

extern long  __wrap_shmem_long_swap(long * a1, long a2, int a3) ;
extern long  __real_shmem_long_swap(long * a1, long a2, int a3)  {

  long retval;
  typedef long (*shmem_long_swap_t)(long * a1, long a2, int a3);
  shmem_long_swap_t shmem_long_swap_handle = (shmem_long_swap_t)get_function_handle("shmem_long_swap");
  retval  =  shmem_long_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern long  shmem_long_swap(long * a1, long a2, int a3) {
   __wrap_shmem_long_swap(a1, a2, a3);
}

extern long shmem_long_swap_(long * a1, long a2, int a3)
{
   __wrap_shmem_long_swap(a1, a2, a3);
}

extern long shmem_long_swap__(long * a1, long a2, int a3)
{
   __wrap_shmem_long_swap(a1, a2, a3);
}

extern long SHMEM_LONG_SWAP_(long * a1, long a2, int a3)
{
   __wrap_shmem_long_swap(a1, a2, a3);
}

extern long SHMEM_LONG_SWAP__(long * a1, long a2, int a3)
{
   __wrap_shmem_long_swap(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_swap
 **********************************************************/

extern long long  __wrap_shmem_longlong_swap(long long * a1, long long a2, int a3) ;
extern long long  __real_shmem_longlong_swap(long long * a1, long long a2, int a3)  {

  long long retval;
  typedef long long (*shmem_longlong_swap_t)(long long * a1, long long a2, int a3);
  shmem_longlong_swap_t shmem_longlong_swap_handle = (shmem_longlong_swap_t)get_function_handle("shmem_longlong_swap");
  retval  =  shmem_longlong_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern long long  shmem_longlong_swap(long long * a1, long long a2, int a3) {
   __wrap_shmem_longlong_swap(a1, a2, a3);
}

extern long long shmem_longlong_swap_(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_swap(a1, a2, a3);
}

extern long long shmem_longlong_swap__(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_swap(a1, a2, a3);
}

extern long long SHMEM_LONGLONG_SWAP_(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_swap(a1, a2, a3);
}

extern long long SHMEM_LONGLONG_SWAP__(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_swap(a1, a2, a3);
}


/**********************************************************
   shmem_float_swap
 **********************************************************/

extern float  __wrap_shmem_float_swap(float * a1, float a2, int a3) ;
extern float  __real_shmem_float_swap(float * a1, float a2, int a3)  {

  float retval;
  typedef float (*shmem_float_swap_t)(float * a1, float a2, int a3);
  shmem_float_swap_t shmem_float_swap_handle = (shmem_float_swap_t)get_function_handle("shmem_float_swap");
  retval  =  shmem_float_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern float  shmem_float_swap(float * a1, float a2, int a3) {
   __wrap_shmem_float_swap(a1, a2, a3);
}

extern float shmem_float_swap_(float * a1, float a2, int a3)
{
   __wrap_shmem_float_swap(a1, a2, a3);
}

extern float shmem_float_swap__(float * a1, float a2, int a3)
{
   __wrap_shmem_float_swap(a1, a2, a3);
}

extern float SHMEM_FLOAT_SWAP_(float * a1, float a2, int a3)
{
   __wrap_shmem_float_swap(a1, a2, a3);
}

extern float SHMEM_FLOAT_SWAP__(float * a1, float a2, int a3)
{
   __wrap_shmem_float_swap(a1, a2, a3);
}


/**********************************************************
   shmem_double_swap
 **********************************************************/

extern double  __wrap_shmem_double_swap(double * a1, double a2, int a3) ;
extern double  __real_shmem_double_swap(double * a1, double a2, int a3)  {

  double retval;
  typedef double (*shmem_double_swap_t)(double * a1, double a2, int a3);
  shmem_double_swap_t shmem_double_swap_handle = (shmem_double_swap_t)get_function_handle("shmem_double_swap");
  retval  =  shmem_double_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern double  shmem_double_swap(double * a1, double a2, int a3) {
   __wrap_shmem_double_swap(a1, a2, a3);
}

extern double shmem_double_swap_(double * a1, double a2, int a3)
{
   __wrap_shmem_double_swap(a1, a2, a3);
}

extern double shmem_double_swap__(double * a1, double a2, int a3)
{
   __wrap_shmem_double_swap(a1, a2, a3);
}

extern double SHMEM_DOUBLE_SWAP_(double * a1, double a2, int a3)
{
   __wrap_shmem_double_swap(a1, a2, a3);
}

extern double SHMEM_DOUBLE_SWAP__(double * a1, double a2, int a3)
{
   __wrap_shmem_double_swap(a1, a2, a3);
}


/**********************************************************
   shmem_short_swap_nb
 **********************************************************/

extern void  __wrap_shmem_short_swap_nb(short * a1, short * a2, short a3, int a4, void ** a5) ;
extern void  __real_shmem_short_swap_nb(short * a1, short * a2, short a3, int a4, void ** a5)  {

  typedef void (*shmem_short_swap_nb_t)(short * a1, short * a2, short a3, int a4, void ** a5);
  shmem_short_swap_nb_t shmem_short_swap_nb_handle = (shmem_short_swap_nb_t)get_function_handle("shmem_short_swap_nb");
  shmem_short_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_short_swap_nb(short * a1, short * a2, short a3, int a4, void ** a5) {
   __wrap_shmem_short_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_short_swap_nb_(short * a1, short * a2, short a3, int a4, void ** a5)
{
   __wrap_shmem_short_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_short_swap_nb__(short * a1, short * a2, short a3, int a4, void ** a5)
{
   __wrap_shmem_short_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_SHORT_SWAP_NB_(short * a1, short * a2, short a3, int a4, void ** a5)
{
   __wrap_shmem_short_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_SHORT_SWAP_NB__(short * a1, short * a2, short a3, int a4, void ** a5)
{
   __wrap_shmem_short_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_int_swap_nb
 **********************************************************/

extern void  __wrap_shmem_int_swap_nb(int * a1, int * a2, int a3, int a4, void ** a5) ;
extern void  __real_shmem_int_swap_nb(int * a1, int * a2, int a3, int a4, void ** a5)  {

  typedef void (*shmem_int_swap_nb_t)(int * a1, int * a2, int a3, int a4, void ** a5);
  shmem_int_swap_nb_t shmem_int_swap_nb_handle = (shmem_int_swap_nb_t)get_function_handle("shmem_int_swap_nb");
  shmem_int_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_int_swap_nb(int * a1, int * a2, int a3, int a4, void ** a5) {
   __wrap_shmem_int_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_int_swap_nb_(int * a1, int * a2, int a3, int a4, void ** a5)
{
   __wrap_shmem_int_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_int_swap_nb__(int * a1, int * a2, int a3, int a4, void ** a5)
{
   __wrap_shmem_int_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_INT_SWAP_NB_(int * a1, int * a2, int a3, int a4, void ** a5)
{
   __wrap_shmem_int_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_INT_SWAP_NB__(int * a1, int * a2, int a3, int a4, void ** a5)
{
   __wrap_shmem_int_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_long_swap_nb
 **********************************************************/

extern void  __wrap_shmem_long_swap_nb(long * a1, long * a2, long a3, int a4, void ** a5) ;
extern void  __real_shmem_long_swap_nb(long * a1, long * a2, long a3, int a4, void ** a5)  {

  typedef void (*shmem_long_swap_nb_t)(long * a1, long * a2, long a3, int a4, void ** a5);
  shmem_long_swap_nb_t shmem_long_swap_nb_handle = (shmem_long_swap_nb_t)get_function_handle("shmem_long_swap_nb");
  shmem_long_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_long_swap_nb(long * a1, long * a2, long a3, int a4, void ** a5) {
   __wrap_shmem_long_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_long_swap_nb_(long * a1, long * a2, long a3, int a4, void ** a5)
{
   __wrap_shmem_long_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_long_swap_nb__(long * a1, long * a2, long a3, int a4, void ** a5)
{
   __wrap_shmem_long_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONG_SWAP_NB_(long * a1, long * a2, long a3, int a4, void ** a5)
{
   __wrap_shmem_long_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONG_SWAP_NB__(long * a1, long * a2, long a3, int a4, void ** a5)
{
   __wrap_shmem_long_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_longlong_swap_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_swap_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) ;
extern void  __real_shmem_longlong_swap_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5)  {

  typedef void (*shmem_longlong_swap_nb_t)(long long * a1, long long * a2, long long a3, int a4, void ** a5);
  shmem_longlong_swap_nb_t shmem_longlong_swap_nb_handle = (shmem_longlong_swap_nb_t)get_function_handle("shmem_longlong_swap_nb");
  shmem_longlong_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_longlong_swap_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) {
   __wrap_shmem_longlong_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_longlong_swap_nb_(long long * a1, long long * a2, long long a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_longlong_swap_nb__(long long * a1, long long * a2, long long a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONGLONG_SWAP_NB_(long long * a1, long long * a2, long long a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONGLONG_SWAP_NB__(long long * a1, long long * a2, long long a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float_swap_nb
 **********************************************************/

extern void  __wrap_shmem_float_swap_nb(float * a1, float * a2, float a3, int a4, void ** a5) ;
extern void  __real_shmem_float_swap_nb(float * a1, float * a2, float a3, int a4, void ** a5)  {

  typedef void (*shmem_float_swap_nb_t)(float * a1, float * a2, float a3, int a4, void ** a5);
  shmem_float_swap_nb_t shmem_float_swap_nb_handle = (shmem_float_swap_nb_t)get_function_handle("shmem_float_swap_nb");
  shmem_float_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float_swap_nb(float * a1, float * a2, float a3, int a4, void ** a5) {
   __wrap_shmem_float_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float_swap_nb_(float * a1, float * a2, float a3, int a4, void ** a5)
{
   __wrap_shmem_float_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_float_swap_nb__(float * a1, float * a2, float a3, int a4, void ** a5)
{
   __wrap_shmem_float_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT_SWAP_NB_(float * a1, float * a2, float a3, int a4, void ** a5)
{
   __wrap_shmem_float_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_FLOAT_SWAP_NB__(float * a1, float * a2, float a3, int a4, void ** a5)
{
   __wrap_shmem_float_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_double_swap_nb
 **********************************************************/

extern void  __wrap_shmem_double_swap_nb(double * a1, double * a2, double a3, int a4, void ** a5) ;
extern void  __real_shmem_double_swap_nb(double * a1, double * a2, double a3, int a4, void ** a5)  {

  typedef void (*shmem_double_swap_nb_t)(double * a1, double * a2, double a3, int a4, void ** a5);
  shmem_double_swap_nb_t shmem_double_swap_nb_handle = (shmem_double_swap_nb_t)get_function_handle("shmem_double_swap_nb");
  shmem_double_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_double_swap_nb(double * a1, double * a2, double a3, int a4, void ** a5) {
   __wrap_shmem_double_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_double_swap_nb_(double * a1, double * a2, double a3, int a4, void ** a5)
{
   __wrap_shmem_double_swap_nb(a1, a2, a3, a4, a5);
}

extern void shmem_double_swap_nb__(double * a1, double * a2, double a3, int a4, void ** a5)
{
   __wrap_shmem_double_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_DOUBLE_SWAP_NB_(double * a1, double * a2, double a3, int a4, void ** a5)
{
   __wrap_shmem_double_swap_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_DOUBLE_SWAP_NB__(double * a1, double * a2, double a3, int a4, void ** a5)
{
   __wrap_shmem_double_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_short_cswap
 **********************************************************/

extern short  __wrap_shmem_short_cswap(short * a1, short a2, short a3, int a4) ;
extern short  __real_shmem_short_cswap(short * a1, short a2, short a3, int a4)  {

  short retval;
  typedef short (*shmem_short_cswap_t)(short * a1, short a2, short a3, int a4);
  shmem_short_cswap_t shmem_short_cswap_handle = (shmem_short_cswap_t)get_function_handle("shmem_short_cswap");
  retval  =  shmem_short_cswap_handle ( a1,  a2,  a3,  a4);
  return retval;

}

extern short  shmem_short_cswap(short * a1, short a2, short a3, int a4) {
   __wrap_shmem_short_cswap(a1, a2, a3, a4);
}

extern short shmem_short_cswap_(short * a1, short a2, short a3, int a4)
{
   __wrap_shmem_short_cswap(a1, a2, a3, a4);
}

extern short shmem_short_cswap__(short * a1, short a2, short a3, int a4)
{
   __wrap_shmem_short_cswap(a1, a2, a3, a4);
}

extern short SHMEM_SHORT_CSWAP_(short * a1, short a2, short a3, int a4)
{
   __wrap_shmem_short_cswap(a1, a2, a3, a4);
}

extern short SHMEM_SHORT_CSWAP__(short * a1, short a2, short a3, int a4)
{
   __wrap_shmem_short_cswap(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_cswap
 **********************************************************/

extern int  __wrap_shmem_int_cswap(int * a1, int a2, int a3, int a4) ;
extern int  __real_shmem_int_cswap(int * a1, int a2, int a3, int a4)  {

  int retval;
  typedef int (*shmem_int_cswap_t)(int * a1, int a2, int a3, int a4);
  shmem_int_cswap_t shmem_int_cswap_handle = (shmem_int_cswap_t)get_function_handle("shmem_int_cswap");
  retval  =  shmem_int_cswap_handle ( a1,  a2,  a3,  a4);
  return retval;

}

extern int  shmem_int_cswap(int * a1, int a2, int a3, int a4) {
   __wrap_shmem_int_cswap(a1, a2, a3, a4);
}

extern int shmem_int_cswap_(int * a1, int a2, int a3, int a4)
{
   __wrap_shmem_int_cswap(a1, a2, a3, a4);
}

extern int shmem_int_cswap__(int * a1, int a2, int a3, int a4)
{
   __wrap_shmem_int_cswap(a1, a2, a3, a4);
}

extern int SHMEM_INT_CSWAP_(int * a1, int a2, int a3, int a4)
{
   __wrap_shmem_int_cswap(a1, a2, a3, a4);
}

extern int SHMEM_INT_CSWAP__(int * a1, int a2, int a3, int a4)
{
   __wrap_shmem_int_cswap(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_cswap
 **********************************************************/

extern long  __wrap_shmem_long_cswap(long * a1, long a2, long a3, int a4) ;
extern long  __real_shmem_long_cswap(long * a1, long a2, long a3, int a4)  {

  long retval;
  typedef long (*shmem_long_cswap_t)(long * a1, long a2, long a3, int a4);
  shmem_long_cswap_t shmem_long_cswap_handle = (shmem_long_cswap_t)get_function_handle("shmem_long_cswap");
  retval  =  shmem_long_cswap_handle ( a1,  a2,  a3,  a4);
  return retval;

}

extern long  shmem_long_cswap(long * a1, long a2, long a3, int a4) {
   __wrap_shmem_long_cswap(a1, a2, a3, a4);
}

extern long shmem_long_cswap_(long * a1, long a2, long a3, int a4)
{
   __wrap_shmem_long_cswap(a1, a2, a3, a4);
}

extern long shmem_long_cswap__(long * a1, long a2, long a3, int a4)
{
   __wrap_shmem_long_cswap(a1, a2, a3, a4);
}

extern long SHMEM_LONG_CSWAP_(long * a1, long a2, long a3, int a4)
{
   __wrap_shmem_long_cswap(a1, a2, a3, a4);
}

extern long SHMEM_LONG_CSWAP__(long * a1, long a2, long a3, int a4)
{
   __wrap_shmem_long_cswap(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_cswap
 **********************************************************/

extern long long  __wrap_shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4) ;
extern long long  __real_shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4)  {

  long long retval;
  typedef long long (*shmem_longlong_cswap_t)(long long * a1, long long a2, long long a3, int a4);
  shmem_longlong_cswap_t shmem_longlong_cswap_handle = (shmem_longlong_cswap_t)get_function_handle("shmem_longlong_cswap");
  retval  =  shmem_longlong_cswap_handle ( a1,  a2,  a3,  a4);
  return retval;

}

extern long long  shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4) {
   __wrap_shmem_longlong_cswap(a1, a2, a3, a4);
}

extern long long shmem_longlong_cswap_(long long * a1, long long a2, long long a3, int a4)
{
   __wrap_shmem_longlong_cswap(a1, a2, a3, a4);
}

extern long long shmem_longlong_cswap__(long long * a1, long long a2, long long a3, int a4)
{
   __wrap_shmem_longlong_cswap(a1, a2, a3, a4);
}

extern long long SHMEM_LONGLONG_CSWAP_(long long * a1, long long a2, long long a3, int a4)
{
   __wrap_shmem_longlong_cswap(a1, a2, a3, a4);
}

extern long long SHMEM_LONGLONG_CSWAP__(long long * a1, long long a2, long long a3, int a4)
{
   __wrap_shmem_longlong_cswap(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_cswap_nb
 **********************************************************/

extern void  __wrap_shmem_short_cswap_nb(short * a1, short * a2, short a3, short a4, int a5, void ** a6) ;
extern void  __real_shmem_short_cswap_nb(short * a1, short * a2, short a3, short a4, int a5, void ** a6)  {

  typedef void (*shmem_short_cswap_nb_t)(short * a1, short * a2, short a3, short a4, int a5, void ** a6);
  shmem_short_cswap_nb_t shmem_short_cswap_nb_handle = (shmem_short_cswap_nb_t)get_function_handle("shmem_short_cswap_nb");
  shmem_short_cswap_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_short_cswap_nb(short * a1, short * a2, short a3, short a4, int a5, void ** a6) {
   __wrap_shmem_short_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void shmem_short_cswap_nb_(short * a1, short * a2, short a3, short a4, int a5, void ** a6)
{
   __wrap_shmem_short_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void shmem_short_cswap_nb__(short * a1, short * a2, short a3, short a4, int a5, void ** a6)
{
   __wrap_shmem_short_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_SHORT_CSWAP_NB_(short * a1, short * a2, short a3, short a4, int a5, void ** a6)
{
   __wrap_shmem_short_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_SHORT_CSWAP_NB__(short * a1, short * a2, short a3, short a4, int a5, void ** a6)
{
   __wrap_shmem_short_cswap_nb(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_int_cswap_nb
 **********************************************************/

extern void  __wrap_shmem_int_cswap_nb(int * a1, int * a2, int a3, int a4, int a5, void ** a6) ;
extern void  __real_shmem_int_cswap_nb(int * a1, int * a2, int a3, int a4, int a5, void ** a6)  {

  typedef void (*shmem_int_cswap_nb_t)(int * a1, int * a2, int a3, int a4, int a5, void ** a6);
  shmem_int_cswap_nb_t shmem_int_cswap_nb_handle = (shmem_int_cswap_nb_t)get_function_handle("shmem_int_cswap_nb");
  shmem_int_cswap_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_int_cswap_nb(int * a1, int * a2, int a3, int a4, int a5, void ** a6) {
   __wrap_shmem_int_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void shmem_int_cswap_nb_(int * a1, int * a2, int a3, int a4, int a5, void ** a6)
{
   __wrap_shmem_int_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void shmem_int_cswap_nb__(int * a1, int * a2, int a3, int a4, int a5, void ** a6)
{
   __wrap_shmem_int_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_INT_CSWAP_NB_(int * a1, int * a2, int a3, int a4, int a5, void ** a6)
{
   __wrap_shmem_int_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_INT_CSWAP_NB__(int * a1, int * a2, int a3, int a4, int a5, void ** a6)
{
   __wrap_shmem_int_cswap_nb(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_long_cswap_nb
 **********************************************************/

extern void  __wrap_shmem_long_cswap_nb(long * a1, long * a2, long a3, long a4, int a5, void ** a6) ;
extern void  __real_shmem_long_cswap_nb(long * a1, long * a2, long a3, long a4, int a5, void ** a6)  {

  typedef void (*shmem_long_cswap_nb_t)(long * a1, long * a2, long a3, long a4, int a5, void ** a6);
  shmem_long_cswap_nb_t shmem_long_cswap_nb_handle = (shmem_long_cswap_nb_t)get_function_handle("shmem_long_cswap_nb");
  shmem_long_cswap_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_long_cswap_nb(long * a1, long * a2, long a3, long a4, int a5, void ** a6) {
   __wrap_shmem_long_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void shmem_long_cswap_nb_(long * a1, long * a2, long a3, long a4, int a5, void ** a6)
{
   __wrap_shmem_long_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void shmem_long_cswap_nb__(long * a1, long * a2, long a3, long a4, int a5, void ** a6)
{
   __wrap_shmem_long_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONG_CSWAP_NB_(long * a1, long * a2, long a3, long a4, int a5, void ** a6)
{
   __wrap_shmem_long_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONG_CSWAP_NB__(long * a1, long * a2, long a3, long a4, int a5, void ** a6)
{
   __wrap_shmem_long_cswap_nb(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_longlong_cswap_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_cswap_nb(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6) ;
extern void  __real_shmem_longlong_cswap_nb(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6)  {

  typedef void (*shmem_longlong_cswap_nb_t)(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6);
  shmem_longlong_cswap_nb_t shmem_longlong_cswap_nb_handle = (shmem_longlong_cswap_nb_t)get_function_handle("shmem_longlong_cswap_nb");
  shmem_longlong_cswap_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longlong_cswap_nb(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6) {
   __wrap_shmem_longlong_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void shmem_longlong_cswap_nb_(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6)
{
   __wrap_shmem_longlong_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void shmem_longlong_cswap_nb__(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6)
{
   __wrap_shmem_longlong_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONGLONG_CSWAP_NB_(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6)
{
   __wrap_shmem_longlong_cswap_nb(a1, a2, a3, a4, a5, a6);
}

extern void SHMEM_LONGLONG_CSWAP_NB__(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6)
{
   __wrap_shmem_longlong_cswap_nb(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_short_finc
 **********************************************************/

extern short  __wrap_shmem_short_finc(short * a1, int a2) ;
extern short  __real_shmem_short_finc(short * a1, int a2)  {

  short retval;
  typedef short (*shmem_short_finc_t)(short * a1, int a2);
  shmem_short_finc_t shmem_short_finc_handle = (shmem_short_finc_t)get_function_handle("shmem_short_finc");
  retval  =  shmem_short_finc_handle ( a1,  a2);
  return retval;

}

extern short  shmem_short_finc(short * a1, int a2) {
   __wrap_shmem_short_finc(a1, a2);
}

extern short shmem_short_finc_(short * a1, int a2)
{
   __wrap_shmem_short_finc(a1, a2);
}

extern short shmem_short_finc__(short * a1, int a2)
{
   __wrap_shmem_short_finc(a1, a2);
}

extern short SHMEM_SHORT_FINC_(short * a1, int a2)
{
   __wrap_shmem_short_finc(a1, a2);
}

extern short SHMEM_SHORT_FINC__(short * a1, int a2)
{
   __wrap_shmem_short_finc(a1, a2);
}


/**********************************************************
   shmem_int_finc
 **********************************************************/

extern int  __wrap_shmem_int_finc(int * a1, int a2) ;
extern int  __real_shmem_int_finc(int * a1, int a2)  {

  int retval;
  typedef int (*shmem_int_finc_t)(int * a1, int a2);
  shmem_int_finc_t shmem_int_finc_handle = (shmem_int_finc_t)get_function_handle("shmem_int_finc");
  retval  =  shmem_int_finc_handle ( a1,  a2);
  return retval;

}

extern int  shmem_int_finc(int * a1, int a2) {
   __wrap_shmem_int_finc(a1, a2);
}

extern int shmem_int_finc_(int * a1, int a2)
{
   __wrap_shmem_int_finc(a1, a2);
}

extern int shmem_int_finc__(int * a1, int a2)
{
   __wrap_shmem_int_finc(a1, a2);
}

extern int SHMEM_INT_FINC_(int * a1, int a2)
{
   __wrap_shmem_int_finc(a1, a2);
}

extern int SHMEM_INT_FINC__(int * a1, int a2)
{
   __wrap_shmem_int_finc(a1, a2);
}


/**********************************************************
   shmem_long_finc
 **********************************************************/

extern long  __wrap_shmem_long_finc(long * a1, int a2) ;
extern long  __real_shmem_long_finc(long * a1, int a2)  {

  long retval;
  typedef long (*shmem_long_finc_t)(long * a1, int a2);
  shmem_long_finc_t shmem_long_finc_handle = (shmem_long_finc_t)get_function_handle("shmem_long_finc");
  retval  =  shmem_long_finc_handle ( a1,  a2);
  return retval;

}

extern long  shmem_long_finc(long * a1, int a2) {
   __wrap_shmem_long_finc(a1, a2);
}

extern long shmem_long_finc_(long * a1, int a2)
{
   __wrap_shmem_long_finc(a1, a2);
}

extern long shmem_long_finc__(long * a1, int a2)
{
   __wrap_shmem_long_finc(a1, a2);
}

extern long SHMEM_LONG_FINC_(long * a1, int a2)
{
   __wrap_shmem_long_finc(a1, a2);
}

extern long SHMEM_LONG_FINC__(long * a1, int a2)
{
   __wrap_shmem_long_finc(a1, a2);
}


/**********************************************************
   shmem_longlong_finc
 **********************************************************/

extern long long  __wrap_shmem_longlong_finc(long long * a1, int a2) ;
extern long long  __real_shmem_longlong_finc(long long * a1, int a2)  {

  long long retval;
  typedef long long (*shmem_longlong_finc_t)(long long * a1, int a2);
  shmem_longlong_finc_t shmem_longlong_finc_handle = (shmem_longlong_finc_t)get_function_handle("shmem_longlong_finc");
  retval  =  shmem_longlong_finc_handle ( a1,  a2);
  return retval;

}

extern long long  shmem_longlong_finc(long long * a1, int a2) {
   __wrap_shmem_longlong_finc(a1, a2);
}

extern long long shmem_longlong_finc_(long long * a1, int a2)
{
   __wrap_shmem_longlong_finc(a1, a2);
}

extern long long shmem_longlong_finc__(long long * a1, int a2)
{
   __wrap_shmem_longlong_finc(a1, a2);
}

extern long long SHMEM_LONGLONG_FINC_(long long * a1, int a2)
{
   __wrap_shmem_longlong_finc(a1, a2);
}

extern long long SHMEM_LONGLONG_FINC__(long long * a1, int a2)
{
   __wrap_shmem_longlong_finc(a1, a2);
}


/**********************************************************
   shmem_short_finc_nb
 **********************************************************/

extern void  __wrap_shmem_short_finc_nb(short * a1, short * a2, int a3, void ** a4) ;
extern void  __real_shmem_short_finc_nb(short * a1, short * a2, int a3, void ** a4)  {

  typedef void (*shmem_short_finc_nb_t)(short * a1, short * a2, int a3, void ** a4);
  shmem_short_finc_nb_t shmem_short_finc_nb_handle = (shmem_short_finc_nb_t)get_function_handle("shmem_short_finc_nb");
  shmem_short_finc_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_short_finc_nb(short * a1, short * a2, int a3, void ** a4) {
   __wrap_shmem_short_finc_nb(a1, a2, a3, a4);
}

extern void shmem_short_finc_nb_(short * a1, short * a2, int a3, void ** a4)
{
   __wrap_shmem_short_finc_nb(a1, a2, a3, a4);
}

extern void shmem_short_finc_nb__(short * a1, short * a2, int a3, void ** a4)
{
   __wrap_shmem_short_finc_nb(a1, a2, a3, a4);
}

extern void SHMEM_SHORT_FINC_NB_(short * a1, short * a2, int a3, void ** a4)
{
   __wrap_shmem_short_finc_nb(a1, a2, a3, a4);
}

extern void SHMEM_SHORT_FINC_NB__(short * a1, short * a2, int a3, void ** a4)
{
   __wrap_shmem_short_finc_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_finc_nb
 **********************************************************/

extern void  __wrap_shmem_int_finc_nb(int * a1, int * a2, int a3, void ** a4) ;
extern void  __real_shmem_int_finc_nb(int * a1, int * a2, int a3, void ** a4)  {

  typedef void (*shmem_int_finc_nb_t)(int * a1, int * a2, int a3, void ** a4);
  shmem_int_finc_nb_t shmem_int_finc_nb_handle = (shmem_int_finc_nb_t)get_function_handle("shmem_int_finc_nb");
  shmem_int_finc_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_int_finc_nb(int * a1, int * a2, int a3, void ** a4) {
   __wrap_shmem_int_finc_nb(a1, a2, a3, a4);
}

extern void shmem_int_finc_nb_(int * a1, int * a2, int a3, void ** a4)
{
   __wrap_shmem_int_finc_nb(a1, a2, a3, a4);
}

extern void shmem_int_finc_nb__(int * a1, int * a2, int a3, void ** a4)
{
   __wrap_shmem_int_finc_nb(a1, a2, a3, a4);
}

extern void SHMEM_INT_FINC_NB_(int * a1, int * a2, int a3, void ** a4)
{
   __wrap_shmem_int_finc_nb(a1, a2, a3, a4);
}

extern void SHMEM_INT_FINC_NB__(int * a1, int * a2, int a3, void ** a4)
{
   __wrap_shmem_int_finc_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_finc_nb
 **********************************************************/

extern void  __wrap_shmem_long_finc_nb(long * a1, long * a2, int a3, void ** a4) ;
extern void  __real_shmem_long_finc_nb(long * a1, long * a2, int a3, void ** a4)  {

  typedef void (*shmem_long_finc_nb_t)(long * a1, long * a2, int a3, void ** a4);
  shmem_long_finc_nb_t shmem_long_finc_nb_handle = (shmem_long_finc_nb_t)get_function_handle("shmem_long_finc_nb");
  shmem_long_finc_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_long_finc_nb(long * a1, long * a2, int a3, void ** a4) {
   __wrap_shmem_long_finc_nb(a1, a2, a3, a4);
}

extern void shmem_long_finc_nb_(long * a1, long * a2, int a3, void ** a4)
{
   __wrap_shmem_long_finc_nb(a1, a2, a3, a4);
}

extern void shmem_long_finc_nb__(long * a1, long * a2, int a3, void ** a4)
{
   __wrap_shmem_long_finc_nb(a1, a2, a3, a4);
}

extern void SHMEM_LONG_FINC_NB_(long * a1, long * a2, int a3, void ** a4)
{
   __wrap_shmem_long_finc_nb(a1, a2, a3, a4);
}

extern void SHMEM_LONG_FINC_NB__(long * a1, long * a2, int a3, void ** a4)
{
   __wrap_shmem_long_finc_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_finc_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_finc_nb(long long * a1, long long * a2, int a3, void ** a4) ;
extern void  __real_shmem_longlong_finc_nb(long long * a1, long long * a2, int a3, void ** a4)  {

  typedef void (*shmem_longlong_finc_nb_t)(long long * a1, long long * a2, int a3, void ** a4);
  shmem_longlong_finc_nb_t shmem_longlong_finc_nb_handle = (shmem_longlong_finc_nb_t)get_function_handle("shmem_longlong_finc_nb");
  shmem_longlong_finc_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_finc_nb(long long * a1, long long * a2, int a3, void ** a4) {
   __wrap_shmem_longlong_finc_nb(a1, a2, a3, a4);
}

extern void shmem_longlong_finc_nb_(long long * a1, long long * a2, int a3, void ** a4)
{
   __wrap_shmem_longlong_finc_nb(a1, a2, a3, a4);
}

extern void shmem_longlong_finc_nb__(long long * a1, long long * a2, int a3, void ** a4)
{
   __wrap_shmem_longlong_finc_nb(a1, a2, a3, a4);
}

extern void SHMEM_LONGLONG_FINC_NB_(long long * a1, long long * a2, int a3, void ** a4)
{
   __wrap_shmem_longlong_finc_nb(a1, a2, a3, a4);
}

extern void SHMEM_LONGLONG_FINC_NB__(long long * a1, long long * a2, int a3, void ** a4)
{
   __wrap_shmem_longlong_finc_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_inc
 **********************************************************/

extern void  __wrap_shmem_short_inc(short * a1, int a2) ;
extern void  __real_shmem_short_inc(short * a1, int a2)  {

  typedef void (*shmem_short_inc_t)(short * a1, int a2);
  shmem_short_inc_t shmem_short_inc_handle = (shmem_short_inc_t)get_function_handle("shmem_short_inc");
  shmem_short_inc_handle ( a1,  a2);

}

extern void  shmem_short_inc(short * a1, int a2) {
   __wrap_shmem_short_inc(a1, a2);
}

extern void shmem_short_inc_(short * a1, int a2)
{
   __wrap_shmem_short_inc(a1, a2);
}

extern void shmem_short_inc__(short * a1, int a2)
{
   __wrap_shmem_short_inc(a1, a2);
}

extern void SHMEM_SHORT_INC_(short * a1, int a2)
{
   __wrap_shmem_short_inc(a1, a2);
}

extern void SHMEM_SHORT_INC__(short * a1, int a2)
{
   __wrap_shmem_short_inc(a1, a2);
}


/**********************************************************
   shmem_int_inc
 **********************************************************/

extern void  __wrap_shmem_int_inc(int * a1, int a2) ;
extern void  __real_shmem_int_inc(int * a1, int a2)  {

  typedef void (*shmem_int_inc_t)(int * a1, int a2);
  shmem_int_inc_t shmem_int_inc_handle = (shmem_int_inc_t)get_function_handle("shmem_int_inc");
  shmem_int_inc_handle ( a1,  a2);

}

extern void  shmem_int_inc(int * a1, int a2) {
   __wrap_shmem_int_inc(a1, a2);
}

extern void shmem_int_inc_(int * a1, int a2)
{
   __wrap_shmem_int_inc(a1, a2);
}

extern void shmem_int_inc__(int * a1, int a2)
{
   __wrap_shmem_int_inc(a1, a2);
}

extern void SHMEM_INT_INC_(int * a1, int a2)
{
   __wrap_shmem_int_inc(a1, a2);
}

extern void SHMEM_INT_INC__(int * a1, int a2)
{
   __wrap_shmem_int_inc(a1, a2);
}


/**********************************************************
   shmem_long_inc
 **********************************************************/

extern void  __wrap_shmem_long_inc(long * a1, int a2) ;
extern void  __real_shmem_long_inc(long * a1, int a2)  {

  typedef void (*shmem_long_inc_t)(long * a1, int a2);
  shmem_long_inc_t shmem_long_inc_handle = (shmem_long_inc_t)get_function_handle("shmem_long_inc");
  shmem_long_inc_handle ( a1,  a2);

}

extern void  shmem_long_inc(long * a1, int a2) {
   __wrap_shmem_long_inc(a1, a2);
}

extern void shmem_long_inc_(long * a1, int a2)
{
   __wrap_shmem_long_inc(a1, a2);
}

extern void shmem_long_inc__(long * a1, int a2)
{
   __wrap_shmem_long_inc(a1, a2);
}

extern void SHMEM_LONG_INC_(long * a1, int a2)
{
   __wrap_shmem_long_inc(a1, a2);
}

extern void SHMEM_LONG_INC__(long * a1, int a2)
{
   __wrap_shmem_long_inc(a1, a2);
}


/**********************************************************
   shmem_longlong_inc
 **********************************************************/

extern void  __wrap_shmem_longlong_inc(long long * a1, int a2) ;
extern void  __real_shmem_longlong_inc(long long * a1, int a2)  {

  typedef void (*shmem_longlong_inc_t)(long long * a1, int a2);
  shmem_longlong_inc_t shmem_longlong_inc_handle = (shmem_longlong_inc_t)get_function_handle("shmem_longlong_inc");
  shmem_longlong_inc_handle ( a1,  a2);

}

extern void  shmem_longlong_inc(long long * a1, int a2) {
   __wrap_shmem_longlong_inc(a1, a2);
}

extern void shmem_longlong_inc_(long long * a1, int a2)
{
   __wrap_shmem_longlong_inc(a1, a2);
}

extern void shmem_longlong_inc__(long long * a1, int a2)
{
   __wrap_shmem_longlong_inc(a1, a2);
}

extern void SHMEM_LONGLONG_INC_(long long * a1, int a2)
{
   __wrap_shmem_longlong_inc(a1, a2);
}

extern void SHMEM_LONGLONG_INC__(long long * a1, int a2)
{
   __wrap_shmem_longlong_inc(a1, a2);
}


/**********************************************************
   shmem_short_inc_nb
 **********************************************************/

extern void  __wrap_shmem_short_inc_nb(short * a1, int a2, void ** a3) ;
extern void  __real_shmem_short_inc_nb(short * a1, int a2, void ** a3)  {

  typedef void (*shmem_short_inc_nb_t)(short * a1, int a2, void ** a3);
  shmem_short_inc_nb_t shmem_short_inc_nb_handle = (shmem_short_inc_nb_t)get_function_handle("shmem_short_inc_nb");
  shmem_short_inc_nb_handle ( a1,  a2,  a3);

}

extern void  shmem_short_inc_nb(short * a1, int a2, void ** a3) {
   __wrap_shmem_short_inc_nb(a1, a2, a3);
}

extern void shmem_short_inc_nb_(short * a1, int a2, void ** a3)
{
   __wrap_shmem_short_inc_nb(a1, a2, a3);
}

extern void shmem_short_inc_nb__(short * a1, int a2, void ** a3)
{
   __wrap_shmem_short_inc_nb(a1, a2, a3);
}

extern void SHMEM_SHORT_INC_NB_(short * a1, int a2, void ** a3)
{
   __wrap_shmem_short_inc_nb(a1, a2, a3);
}

extern void SHMEM_SHORT_INC_NB__(short * a1, int a2, void ** a3)
{
   __wrap_shmem_short_inc_nb(a1, a2, a3);
}


/**********************************************************
   shmem_int_inc_nb
 **********************************************************/

extern void  __wrap_shmem_int_inc_nb(int * a1, int a2, void ** a3) ;
extern void  __real_shmem_int_inc_nb(int * a1, int a2, void ** a3)  {

  typedef void (*shmem_int_inc_nb_t)(int * a1, int a2, void ** a3);
  shmem_int_inc_nb_t shmem_int_inc_nb_handle = (shmem_int_inc_nb_t)get_function_handle("shmem_int_inc_nb");
  shmem_int_inc_nb_handle ( a1,  a2,  a3);

}

extern void  shmem_int_inc_nb(int * a1, int a2, void ** a3) {
   __wrap_shmem_int_inc_nb(a1, a2, a3);
}

extern void shmem_int_inc_nb_(int * a1, int a2, void ** a3)
{
   __wrap_shmem_int_inc_nb(a1, a2, a3);
}

extern void shmem_int_inc_nb__(int * a1, int a2, void ** a3)
{
   __wrap_shmem_int_inc_nb(a1, a2, a3);
}

extern void SHMEM_INT_INC_NB_(int * a1, int a2, void ** a3)
{
   __wrap_shmem_int_inc_nb(a1, a2, a3);
}

extern void SHMEM_INT_INC_NB__(int * a1, int a2, void ** a3)
{
   __wrap_shmem_int_inc_nb(a1, a2, a3);
}


/**********************************************************
   shmem_long_inc_nb
 **********************************************************/

extern void  __wrap_shmem_long_inc_nb(long * a1, int a2, void ** a3) ;
extern void  __real_shmem_long_inc_nb(long * a1, int a2, void ** a3)  {

  typedef void (*shmem_long_inc_nb_t)(long * a1, int a2, void ** a3);
  shmem_long_inc_nb_t shmem_long_inc_nb_handle = (shmem_long_inc_nb_t)get_function_handle("shmem_long_inc_nb");
  shmem_long_inc_nb_handle ( a1,  a2,  a3);

}

extern void  shmem_long_inc_nb(long * a1, int a2, void ** a3) {
   __wrap_shmem_long_inc_nb(a1, a2, a3);
}

extern void shmem_long_inc_nb_(long * a1, int a2, void ** a3)
{
   __wrap_shmem_long_inc_nb(a1, a2, a3);
}

extern void shmem_long_inc_nb__(long * a1, int a2, void ** a3)
{
   __wrap_shmem_long_inc_nb(a1, a2, a3);
}

extern void SHMEM_LONG_INC_NB_(long * a1, int a2, void ** a3)
{
   __wrap_shmem_long_inc_nb(a1, a2, a3);
}

extern void SHMEM_LONG_INC_NB__(long * a1, int a2, void ** a3)
{
   __wrap_shmem_long_inc_nb(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_inc_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_inc_nb(long long * a1, int a2, void ** a3) ;
extern void  __real_shmem_longlong_inc_nb(long long * a1, int a2, void ** a3)  {

  typedef void (*shmem_longlong_inc_nb_t)(long long * a1, int a2, void ** a3);
  shmem_longlong_inc_nb_t shmem_longlong_inc_nb_handle = (shmem_longlong_inc_nb_t)get_function_handle("shmem_longlong_inc_nb");
  shmem_longlong_inc_nb_handle ( a1,  a2,  a3);

}

extern void  shmem_longlong_inc_nb(long long * a1, int a2, void ** a3) {
   __wrap_shmem_longlong_inc_nb(a1, a2, a3);
}

extern void shmem_longlong_inc_nb_(long long * a1, int a2, void ** a3)
{
   __wrap_shmem_longlong_inc_nb(a1, a2, a3);
}

extern void shmem_longlong_inc_nb__(long long * a1, int a2, void ** a3)
{
   __wrap_shmem_longlong_inc_nb(a1, a2, a3);
}

extern void SHMEM_LONGLONG_INC_NB_(long long * a1, int a2, void ** a3)
{
   __wrap_shmem_longlong_inc_nb(a1, a2, a3);
}

extern void SHMEM_LONGLONG_INC_NB__(long long * a1, int a2, void ** a3)
{
   __wrap_shmem_longlong_inc_nb(a1, a2, a3);
}


/**********************************************************
   shmem_short_fadd
 **********************************************************/

extern short  __wrap_shmem_short_fadd(short * a1, short a2, int a3) ;
extern short  __real_shmem_short_fadd(short * a1, short a2, int a3)  {

  short retval;
  typedef short (*shmem_short_fadd_t)(short * a1, short a2, int a3);
  shmem_short_fadd_t shmem_short_fadd_handle = (shmem_short_fadd_t)get_function_handle("shmem_short_fadd");
  retval  =  shmem_short_fadd_handle ( a1,  a2,  a3);
  return retval;

}

extern short  shmem_short_fadd(short * a1, short a2, int a3) {
   __wrap_shmem_short_fadd(a1, a2, a3);
}

extern short shmem_short_fadd_(short * a1, short a2, int a3)
{
   __wrap_shmem_short_fadd(a1, a2, a3);
}

extern short shmem_short_fadd__(short * a1, short a2, int a3)
{
   __wrap_shmem_short_fadd(a1, a2, a3);
}

extern short SHMEM_SHORT_FADD_(short * a1, short a2, int a3)
{
   __wrap_shmem_short_fadd(a1, a2, a3);
}

extern short SHMEM_SHORT_FADD__(short * a1, short a2, int a3)
{
   __wrap_shmem_short_fadd(a1, a2, a3);
}


/**********************************************************
   shmem_int_fadd
 **********************************************************/

extern int  __wrap_shmem_int_fadd(int * a1, int a2, int a3) ;
extern int  __real_shmem_int_fadd(int * a1, int a2, int a3)  {

  int retval;
  typedef int (*shmem_int_fadd_t)(int * a1, int a2, int a3);
  shmem_int_fadd_t shmem_int_fadd_handle = (shmem_int_fadd_t)get_function_handle("shmem_int_fadd");
  retval  =  shmem_int_fadd_handle ( a1,  a2,  a3);
  return retval;

}

extern int  shmem_int_fadd(int * a1, int a2, int a3) {
   __wrap_shmem_int_fadd(a1, a2, a3);
}

extern int shmem_int_fadd_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_fadd(a1, a2, a3);
}

extern int shmem_int_fadd__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_fadd(a1, a2, a3);
}

extern int SHMEM_INT_FADD_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_fadd(a1, a2, a3);
}

extern int SHMEM_INT_FADD__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_fadd(a1, a2, a3);
}


/**********************************************************
   shmem_long_fadd
 **********************************************************/

extern long  __wrap_shmem_long_fadd(long * a1, long a2, int a3) ;
extern long  __real_shmem_long_fadd(long * a1, long a2, int a3)  {

  long retval;
  typedef long (*shmem_long_fadd_t)(long * a1, long a2, int a3);
  shmem_long_fadd_t shmem_long_fadd_handle = (shmem_long_fadd_t)get_function_handle("shmem_long_fadd");
  retval  =  shmem_long_fadd_handle ( a1,  a2,  a3);
  return retval;

}

extern long  shmem_long_fadd(long * a1, long a2, int a3) {
   __wrap_shmem_long_fadd(a1, a2, a3);
}

extern long shmem_long_fadd_(long * a1, long a2, int a3)
{
   __wrap_shmem_long_fadd(a1, a2, a3);
}

extern long shmem_long_fadd__(long * a1, long a2, int a3)
{
   __wrap_shmem_long_fadd(a1, a2, a3);
}

extern long SHMEM_LONG_FADD_(long * a1, long a2, int a3)
{
   __wrap_shmem_long_fadd(a1, a2, a3);
}

extern long SHMEM_LONG_FADD__(long * a1, long a2, int a3)
{
   __wrap_shmem_long_fadd(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_fadd
 **********************************************************/

extern long long  __wrap_shmem_longlong_fadd(long long * a1, long long a2, int a3) ;
extern long long  __real_shmem_longlong_fadd(long long * a1, long long a2, int a3)  {

  long long retval;
  typedef long long (*shmem_longlong_fadd_t)(long long * a1, long long a2, int a3);
  shmem_longlong_fadd_t shmem_longlong_fadd_handle = (shmem_longlong_fadd_t)get_function_handle("shmem_longlong_fadd");
  retval  =  shmem_longlong_fadd_handle ( a1,  a2,  a3);
  return retval;

}

extern long long  shmem_longlong_fadd(long long * a1, long long a2, int a3) {
   __wrap_shmem_longlong_fadd(a1, a2, a3);
}

extern long long shmem_longlong_fadd_(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_fadd(a1, a2, a3);
}

extern long long shmem_longlong_fadd__(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_fadd(a1, a2, a3);
}

extern long long SHMEM_LONGLONG_FADD_(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_fadd(a1, a2, a3);
}

extern long long SHMEM_LONGLONG_FADD__(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_fadd(a1, a2, a3);
}


/**********************************************************
   shmem_short_fadd_nb
 **********************************************************/

extern void  __wrap_shmem_short_fadd_nb(short * a1, short * a2, short a3, int a4, void ** a5) ;
extern void  __real_shmem_short_fadd_nb(short * a1, short * a2, short a3, int a4, void ** a5)  {

  typedef void (*shmem_short_fadd_nb_t)(short * a1, short * a2, short a3, int a4, void ** a5);
  shmem_short_fadd_nb_t shmem_short_fadd_nb_handle = (shmem_short_fadd_nb_t)get_function_handle("shmem_short_fadd_nb");
  shmem_short_fadd_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_short_fadd_nb(short * a1, short * a2, short a3, int a4, void ** a5) {
   __wrap_shmem_short_fadd_nb(a1, a2, a3, a4, a5);
}

extern void shmem_short_fadd_nb_(short * a1, short * a2, short a3, int a4, void ** a5)
{
   __wrap_shmem_short_fadd_nb(a1, a2, a3, a4, a5);
}

extern void shmem_short_fadd_nb__(short * a1, short * a2, short a3, int a4, void ** a5)
{
   __wrap_shmem_short_fadd_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_SHORT_FADD_NB_(short * a1, short * a2, short a3, int a4, void ** a5)
{
   __wrap_shmem_short_fadd_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_SHORT_FADD_NB__(short * a1, short * a2, short a3, int a4, void ** a5)
{
   __wrap_shmem_short_fadd_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_int_fadd_nb
 **********************************************************/

extern void  __wrap_shmem_int_fadd_nb(int * a1, int * a2, int a3, int a4, void ** a5) ;
extern void  __real_shmem_int_fadd_nb(int * a1, int * a2, int a3, int a4, void ** a5)  {

  typedef void (*shmem_int_fadd_nb_t)(int * a1, int * a2, int a3, int a4, void ** a5);
  shmem_int_fadd_nb_t shmem_int_fadd_nb_handle = (shmem_int_fadd_nb_t)get_function_handle("shmem_int_fadd_nb");
  shmem_int_fadd_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_int_fadd_nb(int * a1, int * a2, int a3, int a4, void ** a5) {
   __wrap_shmem_int_fadd_nb(a1, a2, a3, a4, a5);
}

extern void shmem_int_fadd_nb_(int * a1, int * a2, int a3, int a4, void ** a5)
{
   __wrap_shmem_int_fadd_nb(a1, a2, a3, a4, a5);
}

extern void shmem_int_fadd_nb__(int * a1, int * a2, int a3, int a4, void ** a5)
{
   __wrap_shmem_int_fadd_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_INT_FADD_NB_(int * a1, int * a2, int a3, int a4, void ** a5)
{
   __wrap_shmem_int_fadd_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_INT_FADD_NB__(int * a1, int * a2, int a3, int a4, void ** a5)
{
   __wrap_shmem_int_fadd_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_long_fadd_nb
 **********************************************************/

extern void  __wrap_shmem_long_fadd_nb(long * a1, long * a2, long a3, int a4, void ** a5) ;
extern void  __real_shmem_long_fadd_nb(long * a1, long * a2, long a3, int a4, void ** a5)  {

  typedef void (*shmem_long_fadd_nb_t)(long * a1, long * a2, long a3, int a4, void ** a5);
  shmem_long_fadd_nb_t shmem_long_fadd_nb_handle = (shmem_long_fadd_nb_t)get_function_handle("shmem_long_fadd_nb");
  shmem_long_fadd_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_long_fadd_nb(long * a1, long * a2, long a3, int a4, void ** a5) {
   __wrap_shmem_long_fadd_nb(a1, a2, a3, a4, a5);
}

extern void shmem_long_fadd_nb_(long * a1, long * a2, long a3, int a4, void ** a5)
{
   __wrap_shmem_long_fadd_nb(a1, a2, a3, a4, a5);
}

extern void shmem_long_fadd_nb__(long * a1, long * a2, long a3, int a4, void ** a5)
{
   __wrap_shmem_long_fadd_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONG_FADD_NB_(long * a1, long * a2, long a3, int a4, void ** a5)
{
   __wrap_shmem_long_fadd_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONG_FADD_NB__(long * a1, long * a2, long a3, int a4, void ** a5)
{
   __wrap_shmem_long_fadd_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_longlong_fadd_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_fadd_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) ;
extern void  __real_shmem_longlong_fadd_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5)  {

  typedef void (*shmem_longlong_fadd_nb_t)(long long * a1, long long * a2, long long a3, int a4, void ** a5);
  shmem_longlong_fadd_nb_t shmem_longlong_fadd_nb_handle = (shmem_longlong_fadd_nb_t)get_function_handle("shmem_longlong_fadd_nb");
  shmem_longlong_fadd_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_longlong_fadd_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) {
   __wrap_shmem_longlong_fadd_nb(a1, a2, a3, a4, a5);
}

extern void shmem_longlong_fadd_nb_(long long * a1, long long * a2, long long a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_fadd_nb(a1, a2, a3, a4, a5);
}

extern void shmem_longlong_fadd_nb__(long long * a1, long long * a2, long long a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_fadd_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONGLONG_FADD_NB_(long long * a1, long long * a2, long long a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_fadd_nb(a1, a2, a3, a4, a5);
}

extern void SHMEM_LONGLONG_FADD_NB__(long long * a1, long long * a2, long long a3, int a4, void ** a5)
{
   __wrap_shmem_longlong_fadd_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_short_add
 **********************************************************/

extern void  __wrap_shmem_short_add(short * a1, short a2, int a3) ;
extern void  __real_shmem_short_add(short * a1, short a2, int a3)  {

  typedef void (*shmem_short_add_t)(short * a1, short a2, int a3);
  shmem_short_add_t shmem_short_add_handle = (shmem_short_add_t)get_function_handle("shmem_short_add");
  shmem_short_add_handle ( a1,  a2,  a3);

}

extern void  shmem_short_add(short * a1, short a2, int a3) {
   __wrap_shmem_short_add(a1, a2, a3);
}

extern void shmem_short_add_(short * a1, short a2, int a3)
{
   __wrap_shmem_short_add(a1, a2, a3);
}

extern void shmem_short_add__(short * a1, short a2, int a3)
{
   __wrap_shmem_short_add(a1, a2, a3);
}

extern void SHMEM_SHORT_ADD_(short * a1, short a2, int a3)
{
   __wrap_shmem_short_add(a1, a2, a3);
}

extern void SHMEM_SHORT_ADD__(short * a1, short a2, int a3)
{
   __wrap_shmem_short_add(a1, a2, a3);
}


/**********************************************************
   shmem_int_add
 **********************************************************/

extern void  __wrap_shmem_int_add(int * a1, int a2, int a3) ;
extern void  __real_shmem_int_add(int * a1, int a2, int a3)  {

  typedef void (*shmem_int_add_t)(int * a1, int a2, int a3);
  shmem_int_add_t shmem_int_add_handle = (shmem_int_add_t)get_function_handle("shmem_int_add");
  shmem_int_add_handle ( a1,  a2,  a3);

}

extern void  shmem_int_add(int * a1, int a2, int a3) {
   __wrap_shmem_int_add(a1, a2, a3);
}

extern void shmem_int_add_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_add(a1, a2, a3);
}

extern void shmem_int_add__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_add(a1, a2, a3);
}

extern void SHMEM_INT_ADD_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_add(a1, a2, a3);
}

extern void SHMEM_INT_ADD__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_add(a1, a2, a3);
}


/**********************************************************
   shmem_long_add
 **********************************************************/

extern void  __wrap_shmem_long_add(long * a1, long a2, int a3) ;
extern void  __real_shmem_long_add(long * a1, long a2, int a3)  {

  typedef void (*shmem_long_add_t)(long * a1, long a2, int a3);
  shmem_long_add_t shmem_long_add_handle = (shmem_long_add_t)get_function_handle("shmem_long_add");
  shmem_long_add_handle ( a1,  a2,  a3);

}

extern void  shmem_long_add(long * a1, long a2, int a3) {
   __wrap_shmem_long_add(a1, a2, a3);
}

extern void shmem_long_add_(long * a1, long a2, int a3)
{
   __wrap_shmem_long_add(a1, a2, a3);
}

extern void shmem_long_add__(long * a1, long a2, int a3)
{
   __wrap_shmem_long_add(a1, a2, a3);
}

extern void SHMEM_LONG_ADD_(long * a1, long a2, int a3)
{
   __wrap_shmem_long_add(a1, a2, a3);
}

extern void SHMEM_LONG_ADD__(long * a1, long a2, int a3)
{
   __wrap_shmem_long_add(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_add
 **********************************************************/

extern void  __wrap_shmem_longlong_add(long long * a1, long long a2, int a3) ;
extern void  __real_shmem_longlong_add(long long * a1, long long a2, int a3)  {

  typedef void (*shmem_longlong_add_t)(long long * a1, long long a2, int a3);
  shmem_longlong_add_t shmem_longlong_add_handle = (shmem_longlong_add_t)get_function_handle("shmem_longlong_add");
  shmem_longlong_add_handle ( a1,  a2,  a3);

}

extern void  shmem_longlong_add(long long * a1, long long a2, int a3) {
   __wrap_shmem_longlong_add(a1, a2, a3);
}

extern void shmem_longlong_add_(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_add(a1, a2, a3);
}

extern void shmem_longlong_add__(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_add(a1, a2, a3);
}

extern void SHMEM_LONGLONG_ADD_(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_add(a1, a2, a3);
}

extern void SHMEM_LONGLONG_ADD__(long long * a1, long long a2, int a3)
{
   __wrap_shmem_longlong_add(a1, a2, a3);
}


/**********************************************************
   shmem_short_add_nb
 **********************************************************/

extern void  __wrap_shmem_short_add_nb(short * a1, short a2, int a3, void ** a4) ;
extern void  __real_shmem_short_add_nb(short * a1, short a2, int a3, void ** a4)  {

  typedef void (*shmem_short_add_nb_t)(short * a1, short a2, int a3, void ** a4);
  shmem_short_add_nb_t shmem_short_add_nb_handle = (shmem_short_add_nb_t)get_function_handle("shmem_short_add_nb");
  shmem_short_add_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_short_add_nb(short * a1, short a2, int a3, void ** a4) {
   __wrap_shmem_short_add_nb(a1, a2, a3, a4);
}

extern void shmem_short_add_nb_(short * a1, short a2, int a3, void ** a4)
{
   __wrap_shmem_short_add_nb(a1, a2, a3, a4);
}

extern void shmem_short_add_nb__(short * a1, short a2, int a3, void ** a4)
{
   __wrap_shmem_short_add_nb(a1, a2, a3, a4);
}

extern void SHMEM_SHORT_ADD_NB_(short * a1, short a2, int a3, void ** a4)
{
   __wrap_shmem_short_add_nb(a1, a2, a3, a4);
}

extern void SHMEM_SHORT_ADD_NB__(short * a1, short a2, int a3, void ** a4)
{
   __wrap_shmem_short_add_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_add_nb
 **********************************************************/

extern void  __wrap_shmem_int_add_nb(int * a1, int a2, int a3, void ** a4) ;
extern void  __real_shmem_int_add_nb(int * a1, int a2, int a3, void ** a4)  {

  typedef void (*shmem_int_add_nb_t)(int * a1, int a2, int a3, void ** a4);
  shmem_int_add_nb_t shmem_int_add_nb_handle = (shmem_int_add_nb_t)get_function_handle("shmem_int_add_nb");
  shmem_int_add_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_int_add_nb(int * a1, int a2, int a3, void ** a4) {
   __wrap_shmem_int_add_nb(a1, a2, a3, a4);
}

extern void shmem_int_add_nb_(int * a1, int a2, int a3, void ** a4)
{
   __wrap_shmem_int_add_nb(a1, a2, a3, a4);
}

extern void shmem_int_add_nb__(int * a1, int a2, int a3, void ** a4)
{
   __wrap_shmem_int_add_nb(a1, a2, a3, a4);
}

extern void SHMEM_INT_ADD_NB_(int * a1, int a2, int a3, void ** a4)
{
   __wrap_shmem_int_add_nb(a1, a2, a3, a4);
}

extern void SHMEM_INT_ADD_NB__(int * a1, int a2, int a3, void ** a4)
{
   __wrap_shmem_int_add_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_add_nb
 **********************************************************/

extern void  __wrap_shmem_long_add_nb(long * a1, long a2, int a3, void ** a4) ;
extern void  __real_shmem_long_add_nb(long * a1, long a2, int a3, void ** a4)  {

  typedef void (*shmem_long_add_nb_t)(long * a1, long a2, int a3, void ** a4);
  shmem_long_add_nb_t shmem_long_add_nb_handle = (shmem_long_add_nb_t)get_function_handle("shmem_long_add_nb");
  shmem_long_add_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_long_add_nb(long * a1, long a2, int a3, void ** a4) {
   __wrap_shmem_long_add_nb(a1, a2, a3, a4);
}

extern void shmem_long_add_nb_(long * a1, long a2, int a3, void ** a4)
{
   __wrap_shmem_long_add_nb(a1, a2, a3, a4);
}

extern void shmem_long_add_nb__(long * a1, long a2, int a3, void ** a4)
{
   __wrap_shmem_long_add_nb(a1, a2, a3, a4);
}

extern void SHMEM_LONG_ADD_NB_(long * a1, long a2, int a3, void ** a4)
{
   __wrap_shmem_long_add_nb(a1, a2, a3, a4);
}

extern void SHMEM_LONG_ADD_NB__(long * a1, long a2, int a3, void ** a4)
{
   __wrap_shmem_long_add_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_add_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_add_nb(long long * a1, long long a2, int a3, void ** a4) ;
extern void  __real_shmem_longlong_add_nb(long long * a1, long long a2, int a3, void ** a4)  {

  typedef void (*shmem_longlong_add_nb_t)(long long * a1, long long a2, int a3, void ** a4);
  shmem_longlong_add_nb_t shmem_longlong_add_nb_handle = (shmem_longlong_add_nb_t)get_function_handle("shmem_longlong_add_nb");
  shmem_longlong_add_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_add_nb(long long * a1, long long a2, int a3, void ** a4) {
   __wrap_shmem_longlong_add_nb(a1, a2, a3, a4);
}

extern void shmem_longlong_add_nb_(long long * a1, long long a2, int a3, void ** a4)
{
   __wrap_shmem_longlong_add_nb(a1, a2, a3, a4);
}

extern void shmem_longlong_add_nb__(long long * a1, long long a2, int a3, void ** a4)
{
   __wrap_shmem_longlong_add_nb(a1, a2, a3, a4);
}

extern void SHMEM_LONGLONG_ADD_NB_(long long * a1, long long a2, int a3, void ** a4)
{
   __wrap_shmem_longlong_add_nb(a1, a2, a3, a4);
}

extern void SHMEM_LONGLONG_ADD_NB__(long long * a1, long long a2, int a3, void ** a4)
{
   __wrap_shmem_longlong_add_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_barrier_all
 **********************************************************/

extern void  __wrap_shmem_barrier_all() ;
extern void  __real_shmem_barrier_all()  {

  typedef void (*shmem_barrier_all_t)();
  shmem_barrier_all_t shmem_barrier_all_handle = (shmem_barrier_all_t)get_function_handle("shmem_barrier_all");
  shmem_barrier_all_handle ();

}

extern void  shmem_barrier_all() {
   __wrap_shmem_barrier_all();
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

extern void  __wrap_shmem_barrier(int a1, int a2, int a3, long * a4) ;
extern void  __real_shmem_barrier(int a1, int a2, int a3, long * a4)  {

  typedef void (*shmem_barrier_t)(int a1, int a2, int a3, long * a4);
  shmem_barrier_t shmem_barrier_handle = (shmem_barrier_t)get_function_handle("shmem_barrier");
  shmem_barrier_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_barrier(int a1, int a2, int a3, long * a4) {
   __wrap_shmem_barrier(a1, a2, a3, a4);
}

extern void shmem_barrier_(int a1, int a2, int a3, long * a4)
{
   __wrap_shmem_barrier(a1, a2, a3, a4);
}

extern void shmem_barrier__(int a1, int a2, int a3, long * a4)
{
   __wrap_shmem_barrier(a1, a2, a3, a4);
}

extern void SHMEM_BARRIER_(int a1, int a2, int a3, long * a4)
{
   __wrap_shmem_barrier(a1, a2, a3, a4);
}

extern void SHMEM_BARRIER__(int a1, int a2, int a3, long * a4)
{
   __wrap_shmem_barrier(a1, a2, a3, a4);
}


/**********************************************************
   shmem_team_barrier
 **********************************************************/

extern void  __wrap_shmem_team_barrier(shmem_team_t a1, long * a2) ;
extern void  __real_shmem_team_barrier(shmem_team_t a1, long * a2)  {

  typedef void (*shmem_team_barrier_t)(shmem_team_t a1, long * a2);
  shmem_team_barrier_t shmem_team_barrier_handle = (shmem_team_barrier_t)get_function_handle("shmem_team_barrier");
  shmem_team_barrier_handle ( a1,  a2);

}

extern void  shmem_team_barrier(shmem_team_t a1, long * a2) {
   __wrap_shmem_team_barrier(a1, a2);
}

extern void shmem_team_barrier_(shmem_team_t a1, long * a2)
{
   __wrap_shmem_team_barrier(a1, a2);
}

extern void shmem_team_barrier__(shmem_team_t a1, long * a2)
{
   __wrap_shmem_team_barrier(a1, a2);
}

extern void SHMEM_TEAM_BARRIER_(shmem_team_t a1, long * a2)
{
   __wrap_shmem_team_barrier(a1, a2);
}

extern void SHMEM_TEAM_BARRIER__(shmem_team_t a1, long * a2)
{
   __wrap_shmem_team_barrier(a1, a2);
}


/**********************************************************
   shmem_fence
 **********************************************************/

extern void  __wrap_shmem_fence() ;
extern void  __real_shmem_fence()  {

  typedef void (*shmem_fence_t)();
  shmem_fence_t shmem_fence_handle = (shmem_fence_t)get_function_handle("shmem_fence");
  shmem_fence_handle ();

}

extern void  shmem_fence() {
   __wrap_shmem_fence();
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

extern void  __wrap_shmem_quiet() ;
extern void  __real_shmem_quiet()  {

  typedef void (*shmem_quiet_t)();
  shmem_quiet_t shmem_quiet_handle = (shmem_quiet_t)get_function_handle("shmem_quiet");
  shmem_quiet_handle ();

}

extern void  shmem_quiet() {
   __wrap_shmem_quiet();
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
   shmem_set_lock
 **********************************************************/

extern void  __wrap_shmem_set_lock(long * a1) ;
extern void  __real_shmem_set_lock(long * a1)  {

  typedef void (*shmem_set_lock_t)(long * a1);
  shmem_set_lock_t shmem_set_lock_handle = (shmem_set_lock_t)get_function_handle("shmem_set_lock");
  shmem_set_lock_handle ( a1);

}

extern void  shmem_set_lock(long * a1) {
   __wrap_shmem_set_lock(a1);
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

extern void  __wrap_shmem_clear_lock(long * a1) ;
extern void  __real_shmem_clear_lock(long * a1)  {

  typedef void (*shmem_clear_lock_t)(long * a1);
  shmem_clear_lock_t shmem_clear_lock_handle = (shmem_clear_lock_t)get_function_handle("shmem_clear_lock");
  shmem_clear_lock_handle ( a1);

}

extern void  shmem_clear_lock(long * a1) {
   __wrap_shmem_clear_lock(a1);
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

extern int  __wrap_shmem_test_lock(long * a1) ;
extern int  __real_shmem_test_lock(long * a1)  {

  int retval;
  typedef int (*shmem_test_lock_t)(long * a1);
  shmem_test_lock_t shmem_test_lock_handle = (shmem_test_lock_t)get_function_handle("shmem_test_lock");
  retval  =  shmem_test_lock_handle ( a1);
  return retval;

}

extern int  shmem_test_lock(long * a1) {
   __wrap_shmem_test_lock(a1);
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
   shmem_clear_event
 **********************************************************/

extern void  __wrap_shmem_clear_event(long * a1) ;
extern void  __real_shmem_clear_event(long * a1)  {

  typedef void (*shmem_clear_event_t)(long * a1);
  shmem_clear_event_t shmem_clear_event_handle = (shmem_clear_event_t)get_function_handle("shmem_clear_event");
  shmem_clear_event_handle ( a1);

}

extern void  shmem_clear_event(long * a1) {
   __wrap_shmem_clear_event(a1);
}

extern void shmem_clear_event_(long * a1)
{
   __wrap_shmem_clear_event(a1);
}

extern void shmem_clear_event__(long * a1)
{
   __wrap_shmem_clear_event(a1);
}

extern void SHMEM_CLEAR_EVENT_(long * a1)
{
   __wrap_shmem_clear_event(a1);
}

extern void SHMEM_CLEAR_EVENT__(long * a1)
{
   __wrap_shmem_clear_event(a1);
}


/**********************************************************
   shmem_set_event
 **********************************************************/

extern void  __wrap_shmem_set_event(long * a1) ;
extern void  __real_shmem_set_event(long * a1)  {

  typedef void (*shmem_set_event_t)(long * a1);
  shmem_set_event_t shmem_set_event_handle = (shmem_set_event_t)get_function_handle("shmem_set_event");
  shmem_set_event_handle ( a1);

}

extern void  shmem_set_event(long * a1) {
   __wrap_shmem_set_event(a1);
}

extern void shmem_set_event_(long * a1)
{
   __wrap_shmem_set_event(a1);
}

extern void shmem_set_event__(long * a1)
{
   __wrap_shmem_set_event(a1);
}

extern void SHMEM_SET_EVENT_(long * a1)
{
   __wrap_shmem_set_event(a1);
}

extern void SHMEM_SET_EVENT__(long * a1)
{
   __wrap_shmem_set_event(a1);
}


/**********************************************************
   shmem_test_event
 **********************************************************/

extern int  __wrap_shmem_test_event(long * a1) ;
extern int  __real_shmem_test_event(long * a1)  {

  int retval;
  typedef int (*shmem_test_event_t)(long * a1);
  shmem_test_event_t shmem_test_event_handle = (shmem_test_event_t)get_function_handle("shmem_test_event");
  retval  =  shmem_test_event_handle ( a1);
  return retval;

}

extern int  shmem_test_event(long * a1) {
   __wrap_shmem_test_event(a1);
}

extern int shmem_test_event_(long * a1)
{
   __wrap_shmem_test_event(a1);
}

extern int shmem_test_event__(long * a1)
{
   __wrap_shmem_test_event(a1);
}

extern int SHMEM_TEST_EVENT_(long * a1)
{
   __wrap_shmem_test_event(a1);
}

extern int SHMEM_TEST_EVENT__(long * a1)
{
   __wrap_shmem_test_event(a1);
}


/**********************************************************
   shmem_short_wait
 **********************************************************/

extern void  __wrap_shmem_short_wait(short * a1, short a2) ;
extern void  __real_shmem_short_wait(short * a1, short a2)  {

  typedef void (*shmem_short_wait_t)(short * a1, short a2);
  shmem_short_wait_t shmem_short_wait_handle = (shmem_short_wait_t)get_function_handle("shmem_short_wait");
  shmem_short_wait_handle ( a1,  a2);

}

extern void  shmem_short_wait(short * a1, short a2) {
   __wrap_shmem_short_wait(a1, a2);
}

extern void shmem_short_wait_(short * a1, short a2)
{
   __wrap_shmem_short_wait(a1, a2);
}

extern void shmem_short_wait__(short * a1, short a2)
{
   __wrap_shmem_short_wait(a1, a2);
}

extern void SHMEM_SHORT_WAIT_(short * a1, short a2)
{
   __wrap_shmem_short_wait(a1, a2);
}

extern void SHMEM_SHORT_WAIT__(short * a1, short a2)
{
   __wrap_shmem_short_wait(a1, a2);
}


/**********************************************************
   shmem_int_wait
 **********************************************************/

extern void  __wrap_shmem_int_wait(int * a1, int a2) ;
extern void  __real_shmem_int_wait(int * a1, int a2)  {

  typedef void (*shmem_int_wait_t)(int * a1, int a2);
  shmem_int_wait_t shmem_int_wait_handle = (shmem_int_wait_t)get_function_handle("shmem_int_wait");
  shmem_int_wait_handle ( a1,  a2);

}

extern void  shmem_int_wait(int * a1, int a2) {
   __wrap_shmem_int_wait(a1, a2);
}

extern void shmem_int_wait_(int * a1, int a2)
{
   __wrap_shmem_int_wait(a1, a2);
}

extern void shmem_int_wait__(int * a1, int a2)
{
   __wrap_shmem_int_wait(a1, a2);
}

extern void SHMEM_INT_WAIT_(int * a1, int a2)
{
   __wrap_shmem_int_wait(a1, a2);
}

extern void SHMEM_INT_WAIT__(int * a1, int a2)
{
   __wrap_shmem_int_wait(a1, a2);
}


/**********************************************************
   shmem_long_wait
 **********************************************************/

extern void  __wrap_shmem_long_wait(long * a1, long a2) ;
extern void  __real_shmem_long_wait(long * a1, long a2)  {

  typedef void (*shmem_long_wait_t)(long * a1, long a2);
  shmem_long_wait_t shmem_long_wait_handle = (shmem_long_wait_t)get_function_handle("shmem_long_wait");
  shmem_long_wait_handle ( a1,  a2);

}

extern void  shmem_long_wait(long * a1, long a2) {
   __wrap_shmem_long_wait(a1, a2);
}

extern void shmem_long_wait_(long * a1, long a2)
{
   __wrap_shmem_long_wait(a1, a2);
}

extern void shmem_long_wait__(long * a1, long a2)
{
   __wrap_shmem_long_wait(a1, a2);
}

extern void SHMEM_LONG_WAIT_(long * a1, long a2)
{
   __wrap_shmem_long_wait(a1, a2);
}

extern void SHMEM_LONG_WAIT__(long * a1, long a2)
{
   __wrap_shmem_long_wait(a1, a2);
}


/**********************************************************
   shmem_longlong_wait
 **********************************************************/

extern void  __wrap_shmem_longlong_wait(long long * a1, long long a2) ;
extern void  __real_shmem_longlong_wait(long long * a1, long long a2)  {

  typedef void (*shmem_longlong_wait_t)(long long * a1, long long a2);
  shmem_longlong_wait_t shmem_longlong_wait_handle = (shmem_longlong_wait_t)get_function_handle("shmem_longlong_wait");
  shmem_longlong_wait_handle ( a1,  a2);

}

extern void  shmem_longlong_wait(long long * a1, long long a2) {
   __wrap_shmem_longlong_wait(a1, a2);
}

extern void shmem_longlong_wait_(long long * a1, long long a2)
{
   __wrap_shmem_longlong_wait(a1, a2);
}

extern void shmem_longlong_wait__(long long * a1, long long a2)
{
   __wrap_shmem_longlong_wait(a1, a2);
}

extern void SHMEM_LONGLONG_WAIT_(long long * a1, long long a2)
{
   __wrap_shmem_longlong_wait(a1, a2);
}

extern void SHMEM_LONGLONG_WAIT__(long long * a1, long long a2)
{
   __wrap_shmem_longlong_wait(a1, a2);
}


/**********************************************************
   shmem_short_wait_until
 **********************************************************/

extern void  __wrap_shmem_short_wait_until(short * a1, int a2, short a3) ;
extern void  __real_shmem_short_wait_until(short * a1, int a2, short a3)  {

  typedef void (*shmem_short_wait_until_t)(short * a1, int a2, short a3);
  shmem_short_wait_until_t shmem_short_wait_until_handle = (shmem_short_wait_until_t)get_function_handle("shmem_short_wait_until");
  shmem_short_wait_until_handle ( a1,  a2,  a3);

}

extern void  shmem_short_wait_until(short * a1, int a2, short a3) {
   __wrap_shmem_short_wait_until(a1, a2, a3);
}

extern void shmem_short_wait_until_(short * a1, int a2, short a3)
{
   __wrap_shmem_short_wait_until(a1, a2, a3);
}

extern void shmem_short_wait_until__(short * a1, int a2, short a3)
{
   __wrap_shmem_short_wait_until(a1, a2, a3);
}

extern void SHMEM_SHORT_WAIT_UNTIL_(short * a1, int a2, short a3)
{
   __wrap_shmem_short_wait_until(a1, a2, a3);
}

extern void SHMEM_SHORT_WAIT_UNTIL__(short * a1, int a2, short a3)
{
   __wrap_shmem_short_wait_until(a1, a2, a3);
}


/**********************************************************
   shmem_int_wait_until
 **********************************************************/

extern void  __wrap_shmem_int_wait_until(int * a1, int a2, int a3) ;
extern void  __real_shmem_int_wait_until(int * a1, int a2, int a3)  {

  typedef void (*shmem_int_wait_until_t)(int * a1, int a2, int a3);
  shmem_int_wait_until_t shmem_int_wait_until_handle = (shmem_int_wait_until_t)get_function_handle("shmem_int_wait_until");
  shmem_int_wait_until_handle ( a1,  a2,  a3);

}

extern void  shmem_int_wait_until(int * a1, int a2, int a3) {
   __wrap_shmem_int_wait_until(a1, a2, a3);
}

extern void shmem_int_wait_until_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_wait_until(a1, a2, a3);
}

extern void shmem_int_wait_until__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_wait_until(a1, a2, a3);
}

extern void SHMEM_INT_WAIT_UNTIL_(int * a1, int a2, int a3)
{
   __wrap_shmem_int_wait_until(a1, a2, a3);
}

extern void SHMEM_INT_WAIT_UNTIL__(int * a1, int a2, int a3)
{
   __wrap_shmem_int_wait_until(a1, a2, a3);
}


/**********************************************************
   shmem_long_wait_until
 **********************************************************/

extern void  __wrap_shmem_long_wait_until(long * a1, int a2, long a3) ;
extern void  __real_shmem_long_wait_until(long * a1, int a2, long a3)  {

  typedef void (*shmem_long_wait_until_t)(long * a1, int a2, long a3);
  shmem_long_wait_until_t shmem_long_wait_until_handle = (shmem_long_wait_until_t)get_function_handle("shmem_long_wait_until");
  shmem_long_wait_until_handle ( a1,  a2,  a3);

}

extern void  shmem_long_wait_until(long * a1, int a2, long a3) {
   __wrap_shmem_long_wait_until(a1, a2, a3);
}

extern void shmem_long_wait_until_(long * a1, int a2, long a3)
{
   __wrap_shmem_long_wait_until(a1, a2, a3);
}

extern void shmem_long_wait_until__(long * a1, int a2, long a3)
{
   __wrap_shmem_long_wait_until(a1, a2, a3);
}

extern void SHMEM_LONG_WAIT_UNTIL_(long * a1, int a2, long a3)
{
   __wrap_shmem_long_wait_until(a1, a2, a3);
}

extern void SHMEM_LONG_WAIT_UNTIL__(long * a1, int a2, long a3)
{
   __wrap_shmem_long_wait_until(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_wait_until
 **********************************************************/

extern void  __wrap_shmem_longlong_wait_until(long long * a1, int a2, long long a3) ;
extern void  __real_shmem_longlong_wait_until(long long * a1, int a2, long long a3)  {

  typedef void (*shmem_longlong_wait_until_t)(long long * a1, int a2, long long a3);
  shmem_longlong_wait_until_t shmem_longlong_wait_until_handle = (shmem_longlong_wait_until_t)get_function_handle("shmem_longlong_wait_until");
  shmem_longlong_wait_until_handle ( a1,  a2,  a3);

}

extern void  shmem_longlong_wait_until(long long * a1, int a2, long long a3) {
   __wrap_shmem_longlong_wait_until(a1, a2, a3);
}

extern void shmem_longlong_wait_until_(long long * a1, int a2, long long a3)
{
   __wrap_shmem_longlong_wait_until(a1, a2, a3);
}

extern void shmem_longlong_wait_until__(long long * a1, int a2, long long a3)
{
   __wrap_shmem_longlong_wait_until(a1, a2, a3);
}

extern void SHMEM_LONGLONG_WAIT_UNTIL_(long long * a1, int a2, long long a3)
{
   __wrap_shmem_longlong_wait_until(a1, a2, a3);
}

extern void SHMEM_LONGLONG_WAIT_UNTIL__(long long * a1, int a2, long long a3)
{
   __wrap_shmem_longlong_wait_until(a1, a2, a3);
}


/**********************************************************
   shmem_short_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_sum_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_sum_to_all_t shmem_short_sum_to_all_handle = (shmem_short_sum_to_all_t)get_function_handle("shmem_short_sum_to_all");
  shmem_short_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_sum_to_all_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_sum_to_all__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_SUM_TO_ALL_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_SUM_TO_ALL__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_max_to_all
 **********************************************************/

extern void  __wrap_shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_max_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_max_to_all_t shmem_short_max_to_all_handle = (shmem_short_max_to_all_t)get_function_handle("shmem_short_max_to_all");
  shmem_short_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_max_to_all_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_max_to_all__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_MAX_TO_ALL_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_MAX_TO_ALL__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_min_to_all
 **********************************************************/

extern void  __wrap_shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_min_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_min_to_all_t shmem_short_min_to_all_handle = (shmem_short_min_to_all_t)get_function_handle("shmem_short_min_to_all");
  shmem_short_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_min_to_all_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_min_to_all__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_MIN_TO_ALL_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_MIN_TO_ALL__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_prod_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_prod_to_all_t shmem_short_prod_to_all_handle = (shmem_short_prod_to_all_t)get_function_handle("shmem_short_prod_to_all");
  shmem_short_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_prod_to_all_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_prod_to_all__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_PROD_TO_ALL_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_PROD_TO_ALL__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_and_to_all
 **********************************************************/

extern void  __wrap_shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_and_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_and_to_all_t shmem_short_and_to_all_handle = (shmem_short_and_to_all_t)get_function_handle("shmem_short_and_to_all");
  shmem_short_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_and_to_all_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_and_to_all__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_AND_TO_ALL_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_AND_TO_ALL__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_or_to_all
 **********************************************************/

extern void  __wrap_shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_or_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_or_to_all_t shmem_short_or_to_all_handle = (shmem_short_or_to_all_t)get_function_handle("shmem_short_or_to_all");
  shmem_short_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_or_to_all_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_or_to_all__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_OR_TO_ALL_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_OR_TO_ALL__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_xor_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_xor_to_all_t shmem_short_xor_to_all_handle = (shmem_short_xor_to_all_t)get_function_handle("shmem_short_xor_to_all");
  shmem_short_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_xor_to_all_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_short_xor_to_all__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_XOR_TO_ALL_(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_SHORT_XOR_TO_ALL__(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_sum_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_sum_to_all_t shmem_int_sum_to_all_handle = (shmem_int_sum_to_all_t)get_function_handle("shmem_int_sum_to_all");
  shmem_int_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_sum_to_all_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_sum_to_all__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_SUM_TO_ALL_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_SUM_TO_ALL__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_max_to_all
 **********************************************************/

extern void  __wrap_shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_max_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_max_to_all_t shmem_int_max_to_all_handle = (shmem_int_max_to_all_t)get_function_handle("shmem_int_max_to_all");
  shmem_int_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_max_to_all_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_max_to_all__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_MAX_TO_ALL_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_MAX_TO_ALL__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_min_to_all
 **********************************************************/

extern void  __wrap_shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_min_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_min_to_all_t shmem_int_min_to_all_handle = (shmem_int_min_to_all_t)get_function_handle("shmem_int_min_to_all");
  shmem_int_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_min_to_all_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_min_to_all__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_MIN_TO_ALL_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_MIN_TO_ALL__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_prod_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_prod_to_all_t shmem_int_prod_to_all_handle = (shmem_int_prod_to_all_t)get_function_handle("shmem_int_prod_to_all");
  shmem_int_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_prod_to_all_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_prod_to_all__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_PROD_TO_ALL_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_PROD_TO_ALL__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_and_to_all
 **********************************************************/

extern void  __wrap_shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_and_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_and_to_all_t shmem_int_and_to_all_handle = (shmem_int_and_to_all_t)get_function_handle("shmem_int_and_to_all");
  shmem_int_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_and_to_all_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_and_to_all__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_AND_TO_ALL_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_AND_TO_ALL__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_or_to_all
 **********************************************************/

extern void  __wrap_shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_or_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_or_to_all_t shmem_int_or_to_all_handle = (shmem_int_or_to_all_t)get_function_handle("shmem_int_or_to_all");
  shmem_int_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_or_to_all_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_or_to_all__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_OR_TO_ALL_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_OR_TO_ALL__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_xor_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_xor_to_all_t shmem_int_xor_to_all_handle = (shmem_int_xor_to_all_t)get_function_handle("shmem_int_xor_to_all");
  shmem_int_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_xor_to_all_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_int_xor_to_all__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_XOR_TO_ALL_(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_INT_XOR_TO_ALL__(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_sum_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_sum_to_all_t shmem_long_sum_to_all_handle = (shmem_long_sum_to_all_t)get_function_handle("shmem_long_sum_to_all");
  shmem_long_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_sum_to_all_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_sum_to_all__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_SUM_TO_ALL_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_SUM_TO_ALL__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_max_to_all
 **********************************************************/

extern void  __wrap_shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_max_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_max_to_all_t shmem_long_max_to_all_handle = (shmem_long_max_to_all_t)get_function_handle("shmem_long_max_to_all");
  shmem_long_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_max_to_all_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_max_to_all__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_MAX_TO_ALL_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_MAX_TO_ALL__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_min_to_all
 **********************************************************/

extern void  __wrap_shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_min_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_min_to_all_t shmem_long_min_to_all_handle = (shmem_long_min_to_all_t)get_function_handle("shmem_long_min_to_all");
  shmem_long_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_min_to_all_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_min_to_all__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_MIN_TO_ALL_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_MIN_TO_ALL__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_prod_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_prod_to_all_t shmem_long_prod_to_all_handle = (shmem_long_prod_to_all_t)get_function_handle("shmem_long_prod_to_all");
  shmem_long_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_prod_to_all_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_prod_to_all__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_PROD_TO_ALL_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_PROD_TO_ALL__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_and_to_all
 **********************************************************/

extern void  __wrap_shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_and_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_and_to_all_t shmem_long_and_to_all_handle = (shmem_long_and_to_all_t)get_function_handle("shmem_long_and_to_all");
  shmem_long_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_and_to_all_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_and_to_all__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_AND_TO_ALL_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_AND_TO_ALL__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_or_to_all
 **********************************************************/

extern void  __wrap_shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_or_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_or_to_all_t shmem_long_or_to_all_handle = (shmem_long_or_to_all_t)get_function_handle("shmem_long_or_to_all");
  shmem_long_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_or_to_all_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_or_to_all__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_OR_TO_ALL_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_OR_TO_ALL__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_xor_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_xor_to_all_t shmem_long_xor_to_all_handle = (shmem_long_xor_to_all_t)get_function_handle("shmem_long_xor_to_all");
  shmem_long_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_xor_to_all_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_long_xor_to_all__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_XOR_TO_ALL_(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONG_XOR_TO_ALL__(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_sum_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_sum_to_all_t shmem_longlong_sum_to_all_handle = (shmem_longlong_sum_to_all_t)get_function_handle("shmem_longlong_sum_to_all");
  shmem_longlong_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_sum_to_all_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_sum_to_all__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_SUM_TO_ALL_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_SUM_TO_ALL__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_max_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_max_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_max_to_all_t shmem_longlong_max_to_all_handle = (shmem_longlong_max_to_all_t)get_function_handle("shmem_longlong_max_to_all");
  shmem_longlong_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_max_to_all_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_max_to_all__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_MAX_TO_ALL_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_MAX_TO_ALL__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_min_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_min_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_min_to_all_t shmem_longlong_min_to_all_handle = (shmem_longlong_min_to_all_t)get_function_handle("shmem_longlong_min_to_all");
  shmem_longlong_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_min_to_all_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_min_to_all__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_MIN_TO_ALL_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_MIN_TO_ALL__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_prod_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_prod_to_all_t shmem_longlong_prod_to_all_handle = (shmem_longlong_prod_to_all_t)get_function_handle("shmem_longlong_prod_to_all");
  shmem_longlong_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_prod_to_all_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_prod_to_all__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_PROD_TO_ALL_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_PROD_TO_ALL__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_and_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_and_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_and_to_all_t shmem_longlong_and_to_all_handle = (shmem_longlong_and_to_all_t)get_function_handle("shmem_longlong_and_to_all");
  shmem_longlong_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_and_to_all_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_and_to_all__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_AND_TO_ALL_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_AND_TO_ALL__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_or_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_or_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_or_to_all_t shmem_longlong_or_to_all_handle = (shmem_longlong_or_to_all_t)get_function_handle("shmem_longlong_or_to_all");
  shmem_longlong_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_or_to_all_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_or_to_all__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_OR_TO_ALL_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_OR_TO_ALL__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_xor_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_xor_to_all_t shmem_longlong_xor_to_all_handle = (shmem_longlong_xor_to_all_t)get_function_handle("shmem_longlong_xor_to_all");
  shmem_longlong_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_xor_to_all_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_longlong_xor_to_all__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_XOR_TO_ALL_(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LONGLONG_XOR_TO_ALL__(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_sum_to_all_t)(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8);
  shmem_float_sum_to_all_t shmem_float_sum_to_all_handle = (shmem_float_sum_to_all_t)get_function_handle("shmem_float_sum_to_all");
  shmem_float_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float_sum_to_all_(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float_sum_to_all__(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT_SUM_TO_ALL_(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT_SUM_TO_ALL__(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_max_to_all
 **********************************************************/

extern void  __wrap_shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_max_to_all_t)(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8);
  shmem_float_max_to_all_t shmem_float_max_to_all_handle = (shmem_float_max_to_all_t)get_function_handle("shmem_float_max_to_all");
  shmem_float_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float_max_to_all_(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float_max_to_all__(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT_MAX_TO_ALL_(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT_MAX_TO_ALL__(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_min_to_all
 **********************************************************/

extern void  __wrap_shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_min_to_all_t)(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8);
  shmem_float_min_to_all_t shmem_float_min_to_all_handle = (shmem_float_min_to_all_t)get_function_handle("shmem_float_min_to_all");
  shmem_float_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float_min_to_all_(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float_min_to_all__(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT_MIN_TO_ALL_(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT_MIN_TO_ALL__(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_prod_to_all_t)(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8);
  shmem_float_prod_to_all_t shmem_float_prod_to_all_handle = (shmem_float_prod_to_all_t)get_function_handle("shmem_float_prod_to_all");
  shmem_float_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float_prod_to_all_(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float_prod_to_all__(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT_PROD_TO_ALL_(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT_PROD_TO_ALL__(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_sum_to_all_t)(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8);
  shmem_double_sum_to_all_t shmem_double_sum_to_all_handle = (shmem_double_sum_to_all_t)get_function_handle("shmem_double_sum_to_all");
  shmem_double_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_double_sum_to_all_(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_double_sum_to_all__(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_DOUBLE_SUM_TO_ALL_(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_DOUBLE_SUM_TO_ALL__(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_max_to_all
 **********************************************************/

extern void  __wrap_shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_max_to_all_t)(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8);
  shmem_double_max_to_all_t shmem_double_max_to_all_handle = (shmem_double_max_to_all_t)get_function_handle("shmem_double_max_to_all");
  shmem_double_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_double_max_to_all_(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_double_max_to_all__(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_DOUBLE_MAX_TO_ALL_(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_DOUBLE_MAX_TO_ALL__(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_min_to_all
 **********************************************************/

extern void  __wrap_shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_min_to_all_t)(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8);
  shmem_double_min_to_all_t shmem_double_min_to_all_handle = (shmem_double_min_to_all_t)get_function_handle("shmem_double_min_to_all");
  shmem_double_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_double_min_to_all_(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_double_min_to_all__(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_DOUBLE_MIN_TO_ALL_(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_DOUBLE_MIN_TO_ALL__(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_prod_to_all_t)(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8);
  shmem_double_prod_to_all_t shmem_double_prod_to_all_handle = (shmem_double_prod_to_all_t)get_function_handle("shmem_double_prod_to_all");
  shmem_double_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_double_prod_to_all_(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_double_prod_to_all__(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_DOUBLE_PROD_TO_ALL_(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_DOUBLE_PROD_TO_ALL__(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_ld80_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_ld80_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_ld80_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_ld80_sum_to_all_t)(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8);
  shmem_ld80_sum_to_all_t shmem_ld80_sum_to_all_handle = (shmem_ld80_sum_to_all_t)get_function_handle("shmem_ld80_sum_to_all");
  shmem_ld80_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_ld80_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_ld80_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_ld80_sum_to_all_(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_ld80_sum_to_all__(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LD80_SUM_TO_ALL_(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LD80_SUM_TO_ALL__(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_ld80_max_to_all
 **********************************************************/

extern void  __wrap_shmem_ld80_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_ld80_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_ld80_max_to_all_t)(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8);
  shmem_ld80_max_to_all_t shmem_ld80_max_to_all_handle = (shmem_ld80_max_to_all_t)get_function_handle("shmem_ld80_max_to_all");
  shmem_ld80_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_ld80_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_ld80_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_ld80_max_to_all_(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_ld80_max_to_all__(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LD80_MAX_TO_ALL_(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LD80_MAX_TO_ALL__(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_ld80_min_to_all
 **********************************************************/

extern void  __wrap_shmem_ld80_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_ld80_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_ld80_min_to_all_t)(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8);
  shmem_ld80_min_to_all_t shmem_ld80_min_to_all_handle = (shmem_ld80_min_to_all_t)get_function_handle("shmem_ld80_min_to_all");
  shmem_ld80_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_ld80_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_ld80_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_ld80_min_to_all_(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_ld80_min_to_all__(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LD80_MIN_TO_ALL_(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LD80_MIN_TO_ALL__(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_ld80_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_ld80_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_ld80_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_ld80_prod_to_all_t)(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8);
  shmem_ld80_prod_to_all_t shmem_ld80_prod_to_all_handle = (shmem_ld80_prod_to_all_t)get_function_handle("shmem_ld80_prod_to_all");
  shmem_ld80_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_ld80_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_ld80_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_ld80_prod_to_all_(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_ld80_prod_to_all__(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LD80_PROD_TO_ALL_(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_LD80_PROD_TO_ALL__(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float128_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_float128_sum_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __real_shmem_float128_sum_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  typedef void (*shmem_float128_sum_to_all_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8);
  shmem_float128_sum_to_all_t shmem_float128_sum_to_all_handle = (shmem_float128_sum_to_all_t)get_function_handle("shmem_float128_sum_to_all");
  shmem_float128_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float128_sum_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) {
   __wrap_shmem_float128_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float128_sum_to_all_(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float128_sum_to_all__(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT128_SUM_TO_ALL_(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT128_SUM_TO_ALL__(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float128_max_to_all
 **********************************************************/

extern void  __wrap_shmem_float128_max_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __real_shmem_float128_max_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  typedef void (*shmem_float128_max_to_all_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8);
  shmem_float128_max_to_all_t shmem_float128_max_to_all_handle = (shmem_float128_max_to_all_t)get_function_handle("shmem_float128_max_to_all");
  shmem_float128_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float128_max_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) {
   __wrap_shmem_float128_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float128_max_to_all_(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float128_max_to_all__(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT128_MAX_TO_ALL_(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT128_MAX_TO_ALL__(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float128_min_to_all
 **********************************************************/

extern void  __wrap_shmem_float128_min_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __real_shmem_float128_min_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  typedef void (*shmem_float128_min_to_all_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8);
  shmem_float128_min_to_all_t shmem_float128_min_to_all_handle = (shmem_float128_min_to_all_t)get_function_handle("shmem_float128_min_to_all");
  shmem_float128_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float128_min_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) {
   __wrap_shmem_float128_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float128_min_to_all_(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float128_min_to_all__(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT128_MIN_TO_ALL_(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT128_MIN_TO_ALL__(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float128_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_float128_prod_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __real_shmem_float128_prod_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  typedef void (*shmem_float128_prod_to_all_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8);
  shmem_float128_prod_to_all_t shmem_float128_prod_to_all_handle = (shmem_float128_prod_to_all_t)get_function_handle("shmem_float128_prod_to_all");
  shmem_float128_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float128_prod_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) {
   __wrap_shmem_float128_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float128_prod_to_all_(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_float128_prod_to_all__(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT128_PROD_TO_ALL_(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_FLOAT128_PROD_TO_ALL__(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_alltoall
 **********************************************************/

extern void  __wrap_shmem_alltoall(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_alltoall(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_alltoall_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  shmem_alltoall_t shmem_alltoall_handle = (shmem_alltoall_t)get_function_handle("shmem_alltoall");
  shmem_alltoall_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_alltoall(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_alltoall(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_alltoall_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_alltoall(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_alltoall__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_alltoall(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_ALLTOALL_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_alltoall(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_ALLTOALL__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_alltoall(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_team_alltoall
 **********************************************************/

extern void  __wrap_shmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) ;
extern void  __real_shmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)  {

  typedef void (*shmem_team_alltoall_t)(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5);
  shmem_team_alltoall_t shmem_team_alltoall_handle = (shmem_team_alltoall_t)get_function_handle("shmem_team_alltoall");
  shmem_team_alltoall_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) {
   __wrap_shmem_team_alltoall(a1, a2, a3, a4, a5);
}

extern void shmem_team_alltoall_(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)
{
   __wrap_shmem_team_alltoall(a1, a2, a3, a4, a5);
}

extern void shmem_team_alltoall__(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)
{
   __wrap_shmem_team_alltoall(a1, a2, a3, a4, a5);
}

extern void SHMEM_TEAM_ALLTOALL_(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)
{
   __wrap_shmem_team_alltoall(a1, a2, a3, a4, a5);
}

extern void SHMEM_TEAM_ALLTOALL__(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)
{
   __wrap_shmem_team_alltoall(a1, a2, a3, a4, a5);
}


/**********************************************************
   pshmem_team_alltoall
 **********************************************************/

extern void  __wrap_pshmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) ;
extern void  __real_pshmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)  {

  typedef void (*pshmem_team_alltoall_t)(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5);
  pshmem_team_alltoall_t pshmem_team_alltoall_handle = (pshmem_team_alltoall_t)get_function_handle("pshmem_team_alltoall");
  pshmem_team_alltoall_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  pshmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) {
   __wrap_pshmem_team_alltoall(a1, a2, a3, a4, a5);
}

extern void pshmem_team_alltoall_(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)
{
   __wrap_pshmem_team_alltoall(a1, a2, a3, a4, a5);
}

extern void pshmem_team_alltoall__(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)
{
   __wrap_pshmem_team_alltoall(a1, a2, a3, a4, a5);
}

extern void PSHMEM_TEAM_ALLTOALL_(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)
{
   __wrap_pshmem_team_alltoall(a1, a2, a3, a4, a5);
}

extern void PSHMEM_TEAM_ALLTOALL__(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)
{
   __wrap_pshmem_team_alltoall(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_alltoallv
 **********************************************************/

extern void  __wrap_shmem_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) ;
extern void  __real_shmem_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)  {

  typedef void (*shmem_alltoallv_t)(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10);
  shmem_alltoallv_t shmem_alltoallv_handle = (shmem_alltoallv_t)get_function_handle("shmem_alltoallv");
  shmem_alltoallv_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);

}

extern void  shmem_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) {
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

extern void shmem_alltoallv_(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)
{
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

extern void shmem_alltoallv__(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)
{
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

extern void SHMEM_ALLTOALLV_(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)
{
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

extern void SHMEM_ALLTOALLV__(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)
{
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}


/**********************************************************
   shmem_team_alltoallv
 **********************************************************/

extern void  __wrap_shmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __real_shmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  typedef void (*shmem_team_alltoallv_t)(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8);
  shmem_team_alltoallv_t shmem_team_alltoallv_handle = (shmem_team_alltoallv_t)get_function_handle("shmem_team_alltoallv");
  shmem_team_alltoallv_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) {
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_team_alltoallv_(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_team_alltoallv__(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_TEAM_ALLTOALLV_(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_TEAM_ALLTOALLV__(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   pshmem_team_alltoallv
 **********************************************************/

extern void  __wrap_pshmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __real_pshmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  typedef void (*pshmem_team_alltoallv_t)(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8);
  pshmem_team_alltoallv_t pshmem_team_alltoallv_handle = (pshmem_team_alltoallv_t)get_function_handle("pshmem_team_alltoallv");
  pshmem_team_alltoallv_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  pshmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) {
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void pshmem_team_alltoallv_(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void pshmem_team_alltoallv__(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void PSHMEM_TEAM_ALLTOALLV_(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void PSHMEM_TEAM_ALLTOALLV__(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_alltoallv_packed
 **********************************************************/

extern void  __wrap_shmem_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) ;
extern void  __real_shmem_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)  {

  typedef void (*shmem_alltoallv_packed_t)(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10);
  shmem_alltoallv_packed_t shmem_alltoallv_packed_handle = (shmem_alltoallv_packed_t)get_function_handle("shmem_alltoallv_packed");
  shmem_alltoallv_packed_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);

}

extern void  shmem_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) {
   __wrap_shmem_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

extern void shmem_alltoallv_packed_(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)
{
   __wrap_shmem_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

extern void shmem_alltoallv_packed__(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)
{
   __wrap_shmem_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

extern void SHMEM_ALLTOALLV_PACKED_(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)
{
   __wrap_shmem_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

extern void SHMEM_ALLTOALLV_PACKED__(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)
{
   __wrap_shmem_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}


/**********************************************************
   shmem_team_alltoallv_packed
 **********************************************************/

extern void  __wrap_shmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __real_shmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  typedef void (*shmem_team_alltoallv_packed_t)(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8);
  shmem_team_alltoallv_packed_t shmem_team_alltoallv_packed_handle = (shmem_team_alltoallv_packed_t)get_function_handle("shmem_team_alltoallv_packed");
  shmem_team_alltoallv_packed_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) {
   __wrap_shmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_team_alltoallv_packed_(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_shmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void shmem_team_alltoallv_packed__(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_shmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_TEAM_ALLTOALLV_PACKED_(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_shmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void SHMEM_TEAM_ALLTOALLV_PACKED__(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_shmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   pshmem_team_alltoallv_packed
 **********************************************************/

extern void  __wrap_pshmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __real_pshmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  typedef void (*pshmem_team_alltoallv_packed_t)(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8);
  pshmem_team_alltoallv_packed_t pshmem_team_alltoallv_packed_handle = (pshmem_team_alltoallv_packed_t)get_function_handle("pshmem_team_alltoallv_packed");
  pshmem_team_alltoallv_packed_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  pshmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) {
   __wrap_pshmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void pshmem_team_alltoallv_packed_(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_pshmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void pshmem_team_alltoallv_packed__(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_pshmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void PSHMEM_TEAM_ALLTOALLV_PACKED_(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_pshmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}

extern void PSHMEM_TEAM_ALLTOALLV_PACKED__(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)
{
   __wrap_pshmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_collect32
 **********************************************************/

extern void  __wrap_shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_collect32_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  shmem_collect32_t shmem_collect32_handle = (shmem_collect32_t)get_function_handle("shmem_collect32");
  shmem_collect32_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_collect32(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_collect32_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_collect32__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_COLLECT32_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_COLLECT32__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_collect64
 **********************************************************/

extern void  __wrap_shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_collect64_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  shmem_collect64_t shmem_collect64_handle = (shmem_collect64_t)get_function_handle("shmem_collect64");
  shmem_collect64_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_collect64(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_collect64_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_collect64__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_COLLECT64_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_COLLECT64__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_fcollect32
 **********************************************************/

extern void  __wrap_shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_fcollect32_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  shmem_fcollect32_t shmem_fcollect32_handle = (shmem_fcollect32_t)get_function_handle("shmem_fcollect32");
  shmem_fcollect32_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_fcollect32(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_fcollect32_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_fcollect32__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_FCOLLECT32_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_FCOLLECT32__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_fcollect64
 **********************************************************/

extern void  __wrap_shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_fcollect64_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  shmem_fcollect64_t shmem_fcollect64_handle = (shmem_fcollect64_t)get_function_handle("shmem_fcollect64");
  shmem_fcollect64_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_fcollect64(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_fcollect64_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, a3, a4, a5, a6, a7);
}

extern void shmem_fcollect64__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_FCOLLECT64_(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, a3, a4, a5, a6, a7);
}

extern void SHMEM_FCOLLECT64__(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_team_split
 **********************************************************/

extern void  __wrap_shmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __real_shmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)  {

  typedef void (*shmem_team_split_t)(shmem_team_t a1, int a2, int a3, shmem_team_t * a4);
  shmem_team_split_t shmem_team_split_handle = (shmem_team_split_t)get_function_handle("shmem_team_split");
  shmem_team_split_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) {
   __wrap_shmem_team_split(a1, a2, a3, a4);
}

extern void shmem_team_split_(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_shmem_team_split(a1, a2, a3, a4);
}

extern void shmem_team_split__(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_shmem_team_split(a1, a2, a3, a4);
}

extern void SHMEM_TEAM_SPLIT_(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_shmem_team_split(a1, a2, a3, a4);
}

extern void SHMEM_TEAM_SPLIT__(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_shmem_team_split(a1, a2, a3, a4);
}


/**********************************************************
   pshmem_team_split
 **********************************************************/

extern void  __wrap_pshmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __real_pshmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)  {

  typedef void (*pshmem_team_split_t)(shmem_team_t a1, int a2, int a3, shmem_team_t * a4);
  pshmem_team_split_t pshmem_team_split_handle = (pshmem_team_split_t)get_function_handle("pshmem_team_split");
  pshmem_team_split_handle ( a1,  a2,  a3,  a4);

}

extern void  pshmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) {
   __wrap_pshmem_team_split(a1, a2, a3, a4);
}

extern void pshmem_team_split_(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_split(a1, a2, a3, a4);
}

extern void pshmem_team_split__(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_split(a1, a2, a3, a4);
}

extern void PSHMEM_TEAM_SPLIT_(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_split(a1, a2, a3, a4);
}

extern void PSHMEM_TEAM_SPLIT__(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_split(a1, a2, a3, a4);
}


/**********************************************************
   shmem_team_create_strided
 **********************************************************/

extern void  __wrap_shmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __real_shmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4)  {

  typedef void (*shmem_team_create_strided_t)(int a1, int a2, int a3, shmem_team_t * a4);
  shmem_team_create_strided_t shmem_team_create_strided_handle = (shmem_team_create_strided_t)get_function_handle("shmem_team_create_strided");
  shmem_team_create_strided_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) {
   __wrap_shmem_team_create_strided(a1, a2, a3, a4);
}

extern void shmem_team_create_strided_(int a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_shmem_team_create_strided(a1, a2, a3, a4);
}

extern void shmem_team_create_strided__(int a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_shmem_team_create_strided(a1, a2, a3, a4);
}

extern void SHMEM_TEAM_CREATE_STRIDED_(int a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_shmem_team_create_strided(a1, a2, a3, a4);
}

extern void SHMEM_TEAM_CREATE_STRIDED__(int a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_shmem_team_create_strided(a1, a2, a3, a4);
}


/**********************************************************
   pshmem_team_create_strided
 **********************************************************/

extern void  __wrap_pshmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __real_pshmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4)  {

  typedef void (*pshmem_team_create_strided_t)(int a1, int a2, int a3, shmem_team_t * a4);
  pshmem_team_create_strided_t pshmem_team_create_strided_handle = (pshmem_team_create_strided_t)get_function_handle("pshmem_team_create_strided");
  pshmem_team_create_strided_handle ( a1,  a2,  a3,  a4);

}

extern void  pshmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) {
   __wrap_pshmem_team_create_strided(a1, a2, a3, a4);
}

extern void pshmem_team_create_strided_(int a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_create_strided(a1, a2, a3, a4);
}

extern void pshmem_team_create_strided__(int a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_create_strided(a1, a2, a3, a4);
}

extern void PSHMEM_TEAM_CREATE_STRIDED_(int a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_create_strided(a1, a2, a3, a4);
}

extern void PSHMEM_TEAM_CREATE_STRIDED__(int a1, int a2, int a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_create_strided(a1, a2, a3, a4);
}


/**********************************************************
   shmem_team_free
 **********************************************************/

extern void  __wrap_shmem_team_free(shmem_team_t * a1) ;
extern void  __real_shmem_team_free(shmem_team_t * a1)  {

  typedef void (*shmem_team_free_t)(shmem_team_t * a1);
  shmem_team_free_t shmem_team_free_handle = (shmem_team_free_t)get_function_handle("shmem_team_free");
  shmem_team_free_handle ( a1);

}

extern void  shmem_team_free(shmem_team_t * a1) {
   __wrap_shmem_team_free(a1);
}

extern void shmem_team_free_(shmem_team_t * a1)
{
   __wrap_shmem_team_free(a1);
}

extern void shmem_team_free__(shmem_team_t * a1)
{
   __wrap_shmem_team_free(a1);
}

extern void SHMEM_TEAM_FREE_(shmem_team_t * a1)
{
   __wrap_shmem_team_free(a1);
}

extern void SHMEM_TEAM_FREE__(shmem_team_t * a1)
{
   __wrap_shmem_team_free(a1);
}


/**********************************************************
   shmem_team_npes
 **********************************************************/

extern int  __wrap_shmem_team_npes(shmem_team_t a1) ;
extern int  __real_shmem_team_npes(shmem_team_t a1)  {

  int retval;
  typedef int (*shmem_team_npes_t)(shmem_team_t a1);
  shmem_team_npes_t shmem_team_npes_handle = (shmem_team_npes_t)get_function_handle("shmem_team_npes");
  retval  =  shmem_team_npes_handle ( a1);
  return retval;

}

extern int  shmem_team_npes(shmem_team_t a1) {
   __wrap_shmem_team_npes(a1);
}

extern int shmem_team_npes_(shmem_team_t a1)
{
   __wrap_shmem_team_npes(a1);
}

extern int shmem_team_npes__(shmem_team_t a1)
{
   __wrap_shmem_team_npes(a1);
}

extern int SHMEM_TEAM_NPES_(shmem_team_t a1)
{
   __wrap_shmem_team_npes(a1);
}

extern int SHMEM_TEAM_NPES__(shmem_team_t a1)
{
   __wrap_shmem_team_npes(a1);
}


/**********************************************************
   shmem_team_mype
 **********************************************************/

extern int  __wrap_shmem_team_mype(shmem_team_t a1) ;
extern int  __real_shmem_team_mype(shmem_team_t a1)  {

  int retval;
  typedef int (*shmem_team_mype_t)(shmem_team_t a1);
  shmem_team_mype_t shmem_team_mype_handle = (shmem_team_mype_t)get_function_handle("shmem_team_mype");
  retval  =  shmem_team_mype_handle ( a1);
  return retval;

}

extern int  shmem_team_mype(shmem_team_t a1) {
   __wrap_shmem_team_mype(a1);
}

extern int shmem_team_mype_(shmem_team_t a1)
{
   __wrap_shmem_team_mype(a1);
}

extern int shmem_team_mype__(shmem_team_t a1)
{
   __wrap_shmem_team_mype(a1);
}

extern int SHMEM_TEAM_MYPE_(shmem_team_t a1)
{
   __wrap_shmem_team_mype(a1);
}

extern int SHMEM_TEAM_MYPE__(shmem_team_t a1)
{
   __wrap_shmem_team_mype(a1);
}


/**********************************************************
   shmem_team_translate_pe
 **********************************************************/

extern int  __wrap_shmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) ;
extern int  __real_shmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3)  {

  int retval;
  typedef int (*shmem_team_translate_pe_t)(shmem_team_t a1, int a2, shmem_team_t a3);
  shmem_team_translate_pe_t shmem_team_translate_pe_handle = (shmem_team_translate_pe_t)get_function_handle("shmem_team_translate_pe");
  retval  =  shmem_team_translate_pe_handle ( a1,  a2,  a3);
  return retval;

}

extern int  shmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) {
   __wrap_shmem_team_translate_pe(a1, a2, a3);
}

extern int shmem_team_translate_pe_(shmem_team_t a1, int a2, shmem_team_t a3)
{
   __wrap_shmem_team_translate_pe(a1, a2, a3);
}

extern int shmem_team_translate_pe__(shmem_team_t a1, int a2, shmem_team_t a3)
{
   __wrap_shmem_team_translate_pe(a1, a2, a3);
}

extern int SHMEM_TEAM_TRANSLATE_PE_(shmem_team_t a1, int a2, shmem_team_t a3)
{
   __wrap_shmem_team_translate_pe(a1, a2, a3);
}

extern int SHMEM_TEAM_TRANSLATE_PE__(shmem_team_t a1, int a2, shmem_team_t a3)
{
   __wrap_shmem_team_translate_pe(a1, a2, a3);
}


/**********************************************************
   pshmem_team_translate_pe
 **********************************************************/

extern int  __wrap_pshmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) ;
extern int  __real_pshmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3)  {

  int retval;
  typedef int (*pshmem_team_translate_pe_t)(shmem_team_t a1, int a2, shmem_team_t a3);
  pshmem_team_translate_pe_t pshmem_team_translate_pe_handle = (pshmem_team_translate_pe_t)get_function_handle("pshmem_team_translate_pe");
  retval  =  pshmem_team_translate_pe_handle ( a1,  a2,  a3);
  return retval;

}

extern int  pshmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) {
   __wrap_pshmem_team_translate_pe(a1, a2, a3);
}

extern int pshmem_team_translate_pe_(shmem_team_t a1, int a2, shmem_team_t a3)
{
   __wrap_pshmem_team_translate_pe(a1, a2, a3);
}

extern int pshmem_team_translate_pe__(shmem_team_t a1, int a2, shmem_team_t a3)
{
   __wrap_pshmem_team_translate_pe(a1, a2, a3);
}

extern int PSHMEM_TEAM_TRANSLATE_PE_(shmem_team_t a1, int a2, shmem_team_t a3)
{
   __wrap_pshmem_team_translate_pe(a1, a2, a3);
}

extern int PSHMEM_TEAM_TRANSLATE_PE__(shmem_team_t a1, int a2, shmem_team_t a3)
{
   __wrap_pshmem_team_translate_pe(a1, a2, a3);
}


/**********************************************************
   shmem_init
 **********************************************************/

extern void  __wrap_shmem_init() ;
extern void  __real_shmem_init()  {

  typedef void (*shmem_init_t)();
  shmem_init_t shmem_init_handle = (shmem_init_t)get_function_handle("shmem_init");
  shmem_init_handle ();

}

extern void  shmem_init() {
   __wrap_shmem_init();
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

extern void  __wrap_shmem_finalize() ;
extern void  __real_shmem_finalize()  {

  typedef void (*shmem_finalize_t)();
  shmem_finalize_t shmem_finalize_handle = (shmem_finalize_t)get_function_handle("shmem_finalize");
  shmem_finalize_handle ();

}

extern void  shmem_finalize() {
   __wrap_shmem_finalize();
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

extern void  __wrap_shmem_global_exit(int a1) ;
extern void  __real_shmem_global_exit(int a1)  {

  typedef void (*shmem_global_exit_t)(int a1);
  shmem_global_exit_t shmem_global_exit_handle = (shmem_global_exit_t)get_function_handle("shmem_global_exit");
  shmem_global_exit_handle ( a1);

}

extern void  shmem_global_exit(int a1) {
   __wrap_shmem_global_exit(a1);
}

extern void shmem_global_exit_(int a1)
{
   __wrap_shmem_global_exit(a1);
}

extern void shmem_global_exit__(int a1)
{
   __wrap_shmem_global_exit(a1);
}

extern void SHMEM_GLOBAL_EXIT_(int a1)
{
   __wrap_shmem_global_exit(a1);
}

extern void SHMEM_GLOBAL_EXIT__(int a1)
{
   __wrap_shmem_global_exit(a1);
}


/**********************************************************
   shmem_n_pes
 **********************************************************/

extern int  __wrap_shmem_n_pes() ;
extern int  __real_shmem_n_pes()  {

  int retval;
  typedef int (*shmem_n_pes_t)();
  shmem_n_pes_t shmem_n_pes_handle = (shmem_n_pes_t)get_function_handle("shmem_n_pes");
  retval  =  shmem_n_pes_handle ();
  return retval;

}

extern int  shmem_n_pes() {
   __wrap_shmem_n_pes();
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
   shmem_my_pe
 **********************************************************/

extern int  __wrap_shmem_my_pe() ;
extern int  __real_shmem_my_pe()  {

  int retval;
  typedef int (*shmem_my_pe_t)();
  shmem_my_pe_t shmem_my_pe_handle = (shmem_my_pe_t)get_function_handle("shmem_my_pe");
  retval  =  shmem_my_pe_handle ();
  return retval;

}

extern int  shmem_my_pe() {
   __wrap_shmem_my_pe();
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
   shmem_pe_accessible
 **********************************************************/

extern int  __wrap_shmem_pe_accessible(int a1) ;
extern int  __real_shmem_pe_accessible(int a1)  {

  int retval;
  typedef int (*shmem_pe_accessible_t)(int a1);
  shmem_pe_accessible_t shmem_pe_accessible_handle = (shmem_pe_accessible_t)get_function_handle("shmem_pe_accessible");
  retval  =  shmem_pe_accessible_handle ( a1);
  return retval;

}

extern int  shmem_pe_accessible(int a1) {
   __wrap_shmem_pe_accessible(a1);
}

extern int shmem_pe_accessible_(int a1)
{
   __wrap_shmem_pe_accessible(a1);
}

extern int shmem_pe_accessible__(int a1)
{
   __wrap_shmem_pe_accessible(a1);
}

extern int SHMEM_PE_ACCESSIBLE_(int a1)
{
   __wrap_shmem_pe_accessible(a1);
}

extern int SHMEM_PE_ACCESSIBLE__(int a1)
{
   __wrap_shmem_pe_accessible(a1);
}


/**********************************************************
   shmem_addr_accessible
 **********************************************************/

extern int  __wrap_shmem_addr_accessible(void * a1, int a2) ;
extern int  __real_shmem_addr_accessible(void * a1, int a2)  {

  int retval;
  typedef int (*shmem_addr_accessible_t)(void * a1, int a2);
  shmem_addr_accessible_t shmem_addr_accessible_handle = (shmem_addr_accessible_t)get_function_handle("shmem_addr_accessible");
  retval  =  shmem_addr_accessible_handle ( a1,  a2);
  return retval;

}

extern int  shmem_addr_accessible(void * a1, int a2) {
   __wrap_shmem_addr_accessible(a1, a2);
}

extern int shmem_addr_accessible_(void * a1, int a2)
{
   __wrap_shmem_addr_accessible(a1, a2);
}

extern int shmem_addr_accessible__(void * a1, int a2)
{
   __wrap_shmem_addr_accessible(a1, a2);
}

extern int SHMEM_ADDR_ACCESSIBLE_(void * a1, int a2)
{
   __wrap_shmem_addr_accessible(a1, a2);
}

extern int SHMEM_ADDR_ACCESSIBLE__(void * a1, int a2)
{
   __wrap_shmem_addr_accessible(a1, a2);
}


/**********************************************************
   shmem_init_thread
 **********************************************************/

extern int  __wrap_shmem_init_thread(int a1) ;
extern int  __real_shmem_init_thread(int a1)  {

  int retval;
  typedef int (*shmem_init_thread_t)(int a1);
  shmem_init_thread_t shmem_init_thread_handle = (shmem_init_thread_t)get_function_handle("shmem_init_thread");
  retval  =  shmem_init_thread_handle ( a1);
  return retval;

}

extern int  shmem_init_thread(int a1) {
   __wrap_shmem_init_thread(a1);
}

extern int shmem_init_thread_(int a1)
{
   __wrap_shmem_init_thread(a1);
}

extern int shmem_init_thread__(int a1)
{
   __wrap_shmem_init_thread(a1);
}

extern int SHMEM_INIT_THREAD_(int a1)
{
   __wrap_shmem_init_thread(a1);
}

extern int SHMEM_INIT_THREAD__(int a1)
{
   __wrap_shmem_init_thread(a1);
}


/**********************************************************
   shmem_query_thread
 **********************************************************/

extern int  __wrap_shmem_query_thread() ;
extern int  __real_shmem_query_thread()  {

  int retval;
  typedef int (*shmem_query_thread_t)();
  shmem_query_thread_t shmem_query_thread_handle = (shmem_query_thread_t)get_function_handle("shmem_query_thread");
  retval  =  shmem_query_thread_handle ();
  return retval;

}

extern int  shmem_query_thread() {
   __wrap_shmem_query_thread();
}

extern int shmem_query_thread_()
{
   __wrap_shmem_query_thread();
}

extern int shmem_query_thread__()
{
   __wrap_shmem_query_thread();
}

extern int SHMEM_QUERY_THREAD_()
{
   __wrap_shmem_query_thread();
}

extern int SHMEM_QUERY_THREAD__()
{
   __wrap_shmem_query_thread();
}


/**********************************************************
   shmem_thread_register
 **********************************************************/

extern void  __wrap_shmem_thread_register() ;
extern void  __real_shmem_thread_register()  {

  typedef void (*shmem_thread_register_t)();
  shmem_thread_register_t shmem_thread_register_handle = (shmem_thread_register_t)get_function_handle("shmem_thread_register");
  shmem_thread_register_handle ();

}

extern void  shmem_thread_register() {
   __wrap_shmem_thread_register();
}

extern void shmem_thread_register_()
{
   __wrap_shmem_thread_register();
}

extern void shmem_thread_register__()
{
   __wrap_shmem_thread_register();
}

extern void SHMEM_THREAD_REGISTER_()
{
   __wrap_shmem_thread_register();
}

extern void SHMEM_THREAD_REGISTER__()
{
   __wrap_shmem_thread_register();
}


/**********************************************************
   shmem_thread_unregister
 **********************************************************/

extern void  __wrap_shmem_thread_unregister() ;
extern void  __real_shmem_thread_unregister()  {

  typedef void (*shmem_thread_unregister_t)();
  shmem_thread_unregister_t shmem_thread_unregister_handle = (shmem_thread_unregister_t)get_function_handle("shmem_thread_unregister");
  shmem_thread_unregister_handle ();

}

extern void  shmem_thread_unregister() {
   __wrap_shmem_thread_unregister();
}

extern void shmem_thread_unregister_()
{
   __wrap_shmem_thread_unregister();
}

extern void shmem_thread_unregister__()
{
   __wrap_shmem_thread_unregister();
}

extern void SHMEM_THREAD_UNREGISTER_()
{
   __wrap_shmem_thread_unregister();
}

extern void SHMEM_THREAD_UNREGISTER__()
{
   __wrap_shmem_thread_unregister();
}


/**********************************************************
   shmem_thread_fence
 **********************************************************/

extern void  __wrap_shmem_thread_fence() ;
extern void  __real_shmem_thread_fence()  {

  typedef void (*shmem_thread_fence_t)();
  shmem_thread_fence_t shmem_thread_fence_handle = (shmem_thread_fence_t)get_function_handle("shmem_thread_fence");
  shmem_thread_fence_handle ();

}

extern void  shmem_thread_fence() {
   __wrap_shmem_thread_fence();
}

extern void shmem_thread_fence_()
{
   __wrap_shmem_thread_fence();
}

extern void shmem_thread_fence__()
{
   __wrap_shmem_thread_fence();
}

extern void SHMEM_THREAD_FENCE_()
{
   __wrap_shmem_thread_fence();
}

extern void SHMEM_THREAD_FENCE__()
{
   __wrap_shmem_thread_fence();
}


/**********************************************************
   shmem_thread_quiet
 **********************************************************/

extern void  __wrap_shmem_thread_quiet() ;
extern void  __real_shmem_thread_quiet()  {

  typedef void (*shmem_thread_quiet_t)();
  shmem_thread_quiet_t shmem_thread_quiet_handle = (shmem_thread_quiet_t)get_function_handle("shmem_thread_quiet");
  shmem_thread_quiet_handle ();

}

extern void  shmem_thread_quiet() {
   __wrap_shmem_thread_quiet();
}

extern void shmem_thread_quiet_()
{
   __wrap_shmem_thread_quiet();
}

extern void shmem_thread_quiet__()
{
   __wrap_shmem_thread_quiet();
}

extern void SHMEM_THREAD_QUIET_()
{
   __wrap_shmem_thread_quiet();
}

extern void SHMEM_THREAD_QUIET__()
{
   __wrap_shmem_thread_quiet();
}


/**********************************************************
   shmem_local_npes
 **********************************************************/

extern int  __wrap_shmem_local_npes() ;
extern int  __real_shmem_local_npes()  {

  int retval;
  typedef int (*shmem_local_npes_t)();
  shmem_local_npes_t shmem_local_npes_handle = (shmem_local_npes_t)get_function_handle("shmem_local_npes");
  retval  =  shmem_local_npes_handle ();
  return retval;

}

extern int  shmem_local_npes() {
   __wrap_shmem_local_npes();
}

extern int shmem_local_npes_()
{
   __wrap_shmem_local_npes();
}

extern int shmem_local_npes__()
{
   __wrap_shmem_local_npes();
}

extern int SHMEM_LOCAL_NPES_()
{
   __wrap_shmem_local_npes();
}

extern int SHMEM_LOCAL_NPES__()
{
   __wrap_shmem_local_npes();
}


/**********************************************************
   shmem_local_pes
 **********************************************************/

extern void  __wrap_shmem_local_pes(int * a1, int a2) ;
extern void  __real_shmem_local_pes(int * a1, int a2)  {

  typedef void (*shmem_local_pes_t)(int * a1, int a2);
  shmem_local_pes_t shmem_local_pes_handle = (shmem_local_pes_t)get_function_handle("shmem_local_pes");
  shmem_local_pes_handle ( a1,  a2);

}

extern void  shmem_local_pes(int * a1, int a2) {
   __wrap_shmem_local_pes(a1, a2);
}

extern void shmem_local_pes_(int * a1, int a2)
{
   __wrap_shmem_local_pes(a1, a2);
}

extern void shmem_local_pes__(int * a1, int a2)
{
   __wrap_shmem_local_pes(a1, a2);
}

extern void SHMEM_LOCAL_PES_(int * a1, int a2)
{
   __wrap_shmem_local_pes(a1, a2);
}

extern void SHMEM_LOCAL_PES__(int * a1, int a2)
{
   __wrap_shmem_local_pes(a1, a2);
}


/**********************************************************
   shmem_set_cache_inv
 **********************************************************/

extern void  __wrap_shmem_set_cache_inv() ;
extern void  __real_shmem_set_cache_inv()  {

  typedef void (*shmem_set_cache_inv_t)();
  shmem_set_cache_inv_t shmem_set_cache_inv_handle = (shmem_set_cache_inv_t)get_function_handle("shmem_set_cache_inv");
  shmem_set_cache_inv_handle ();

}

extern void  shmem_set_cache_inv() {
   __wrap_shmem_set_cache_inv();
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
   shmem_set_cache_line_inv
 **********************************************************/

extern void  __wrap_shmem_set_cache_line_inv(void * a1) ;
extern void  __real_shmem_set_cache_line_inv(void * a1)  {

  typedef void (*shmem_set_cache_line_inv_t)(void * a1);
  shmem_set_cache_line_inv_t shmem_set_cache_line_inv_handle = (shmem_set_cache_line_inv_t)get_function_handle("shmem_set_cache_line_inv");
  shmem_set_cache_line_inv_handle ( a1);

}

extern void  shmem_set_cache_line_inv(void * a1) {
   __wrap_shmem_set_cache_line_inv(a1);
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
   shmem_clear_cache_inv
 **********************************************************/

extern void  __wrap_shmem_clear_cache_inv() ;
extern void  __real_shmem_clear_cache_inv()  {

  typedef void (*shmem_clear_cache_inv_t)();
  shmem_clear_cache_inv_t shmem_clear_cache_inv_handle = (shmem_clear_cache_inv_t)get_function_handle("shmem_clear_cache_inv");
  shmem_clear_cache_inv_handle ();

}

extern void  shmem_clear_cache_inv() {
   __wrap_shmem_clear_cache_inv();
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
   shmem_clear_cache_line_inv
 **********************************************************/

extern void  __wrap_shmem_clear_cache_line_inv(void * a1) ;
extern void  __real_shmem_clear_cache_line_inv(void * a1)  {

  typedef void (*shmem_clear_cache_line_inv_t)(void * a1);
  shmem_clear_cache_line_inv_t shmem_clear_cache_line_inv_handle = (shmem_clear_cache_line_inv_t)get_function_handle("shmem_clear_cache_line_inv");
  shmem_clear_cache_line_inv_handle ( a1);

}

extern void  shmem_clear_cache_line_inv(void * a1) {
   __wrap_shmem_clear_cache_line_inv(a1);
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
   shmem_udcflush
 **********************************************************/

extern void  __wrap_shmem_udcflush() ;
extern void  __real_shmem_udcflush()  {

  typedef void (*shmem_udcflush_t)();
  shmem_udcflush_t shmem_udcflush_handle = (shmem_udcflush_t)get_function_handle("shmem_udcflush");
  shmem_udcflush_handle ();

}

extern void  shmem_udcflush() {
   __wrap_shmem_udcflush();
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

extern void  __wrap_shmem_udcflush_line(void * a1) ;
extern void  __real_shmem_udcflush_line(void * a1)  {

  typedef void (*shmem_udcflush_line_t)(void * a1);
  shmem_udcflush_line_t shmem_udcflush_line_handle = (shmem_udcflush_line_t)get_function_handle("shmem_udcflush_line");
  shmem_udcflush_line_handle ( a1);

}

extern void  shmem_udcflush_line(void * a1) {
   __wrap_shmem_udcflush_line(a1);
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

