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


/**********************************************************
   shmem_longdouble_put
 **********************************************************/

extern void  __wrap_shmem_longdouble_put(long double * a1, const long double * a2, size_t a3, int a4) ;
extern void  __real_shmem_longdouble_put(long double * a1, const long double * a2, size_t a3, int a4)  {

  typedef void (*shmem_longdouble_put_t)(long double * a1, const long double * a2, size_t a3, int a4);
  shmem_longdouble_put_t shmem_longdouble_put_handle = (shmem_longdouble_put_t)get_function_handle("shmem_longdouble_put");
  shmem_longdouble_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longdouble_put(long double * a1, const long double * a2, size_t a3, int a4) {
   __wrap_shmem_longdouble_put(a1, a2, a3, a4);
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


/**********************************************************
   shmem_putmem
 **********************************************************/

extern void  __wrap_shmem_putmem(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_putmem(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_putmem_t)(void * a1, const void * a2, size_t a3, int a4);
  shmem_putmem_t shmem_putmem_handle = (shmem_putmem_t)get_function_handle("shmem_putmem");
  shmem_putmem_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_putmem(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_putmem(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put32
 **********************************************************/

extern void  __wrap_shmem_put32(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put32(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put32_t)(void * a1, const void * a2, size_t a3, int a4);
  shmem_put32_t shmem_put32_handle = (shmem_put32_t)get_function_handle("shmem_put32");
  shmem_put32_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put32(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put32(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put64
 **********************************************************/

extern void  __wrap_shmem_put64(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put64(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put64_t)(void * a1, const void * a2, size_t a3, int a4);
  shmem_put64_t shmem_put64_handle = (shmem_put64_t)get_function_handle("shmem_put64");
  shmem_put64_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put64(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put64(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put128
 **********************************************************/

extern void  __wrap_shmem_put128(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put128(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put128_t)(void * a1, const void * a2, size_t a3, int a4);
  shmem_put128_t shmem_put128_handle = (shmem_put128_t)get_function_handle("shmem_put128");
  shmem_put128_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put128(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put128(a1, a2, a3, a4);
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


/**********************************************************
   shmem_longdouble_get
 **********************************************************/

extern void  __wrap_shmem_longdouble_get(long double * a1, const long double * a2, size_t a3, int a4) ;
extern void  __real_shmem_longdouble_get(long double * a1, const long double * a2, size_t a3, int a4)  {

  typedef void (*shmem_longdouble_get_t)(long double * a1, const long double * a2, size_t a3, int a4);
  shmem_longdouble_get_t shmem_longdouble_get_handle = (shmem_longdouble_get_t)get_function_handle("shmem_longdouble_get");
  shmem_longdouble_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longdouble_get(long double * a1, const long double * a2, size_t a3, int a4) {
   __wrap_shmem_longdouble_get(a1, a2, a3, a4);
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


/**********************************************************
   shmem_getmem
 **********************************************************/

extern void  __wrap_shmem_getmem(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_getmem(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_getmem_t)(void * a1, const void * a2, size_t a3, int a4);
  shmem_getmem_t shmem_getmem_handle = (shmem_getmem_t)get_function_handle("shmem_getmem");
  shmem_getmem_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_getmem(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_getmem(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get32
 **********************************************************/

extern void  __wrap_shmem_get32(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get32(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get32_t)(void * a1, const void * a2, size_t a3, int a4);
  shmem_get32_t shmem_get32_handle = (shmem_get32_t)get_function_handle("shmem_get32");
  shmem_get32_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get32(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get32(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get64
 **********************************************************/

extern void  __wrap_shmem_get64(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get64(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get64_t)(void * a1, const void * a2, size_t a3, int a4);
  shmem_get64_t shmem_get64_handle = (shmem_get64_t)get_function_handle("shmem_get64");
  shmem_get64_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get64(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get64(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get128
 **********************************************************/

extern void  __wrap_shmem_get128(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get128(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get128_t)(void * a1, const void * a2, size_t a3, int a4);
  shmem_get128_t shmem_get128_handle = (shmem_get128_t)get_function_handle("shmem_get128");
  shmem_get128_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get128(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get128(a1, a2, a3, a4);
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


/**********************************************************
   shmem_longdouble_p
 **********************************************************/

extern void  __wrap_shmem_longdouble_p(long double * a1, long double a2, int a3) ;
extern void  __real_shmem_longdouble_p(long double * a1, long double a2, int a3)  {

  typedef void (*shmem_longdouble_p_t)(long double * a1, long double a2, int a3);
  shmem_longdouble_p_t shmem_longdouble_p_handle = (shmem_longdouble_p_t)get_function_handle("shmem_longdouble_p");
  shmem_longdouble_p_handle ( a1,  a2,  a3);

}

extern void  shmem_longdouble_p(long double * a1, long double a2, int a3) {
   __wrap_shmem_longdouble_p(a1, a2, a3);
}


/**********************************************************
   shmem_char_g
 **********************************************************/

extern char  __wrap_shmem_char_g(char * a1, int a2) ;
extern char  __real_shmem_char_g(char * a1, int a2)  {

  char retval;
  typedef char (*shmem_char_g_t)(char * a1, int a2);
  shmem_char_g_t shmem_char_g_handle = (shmem_char_g_t)get_function_handle("shmem_char_g");
  retval  =  shmem_char_g_handle ( a1,  a2);
  return retval;

}

extern char  shmem_char_g(char * a1, int a2) {
   __wrap_shmem_char_g(a1, a2);
}


/**********************************************************
   shmem_short_g
 **********************************************************/

extern short  __wrap_shmem_short_g(short * a1, int a2) ;
extern short  __real_shmem_short_g(short * a1, int a2)  {

  short retval;
  typedef short (*shmem_short_g_t)(short * a1, int a2);
  shmem_short_g_t shmem_short_g_handle = (shmem_short_g_t)get_function_handle("shmem_short_g");
  retval  =  shmem_short_g_handle ( a1,  a2);
  return retval;

}

extern short  shmem_short_g(short * a1, int a2) {
   __wrap_shmem_short_g(a1, a2);
}


/**********************************************************
   shmem_int_g
 **********************************************************/

extern int  __wrap_shmem_int_g(int * a1, int a2) ;
extern int  __real_shmem_int_g(int * a1, int a2)  {

  int retval;
  typedef int (*shmem_int_g_t)(int * a1, int a2);
  shmem_int_g_t shmem_int_g_handle = (shmem_int_g_t)get_function_handle("shmem_int_g");
  retval  =  shmem_int_g_handle ( a1,  a2);
  return retval;

}

extern int  shmem_int_g(int * a1, int a2) {
   __wrap_shmem_int_g(a1, a2);
}


/**********************************************************
   shmem_long_g
 **********************************************************/

extern long  __wrap_shmem_long_g(long * a1, int a2) ;
extern long  __real_shmem_long_g(long * a1, int a2)  {

  long retval;
  typedef long (*shmem_long_g_t)(long * a1, int a2);
  shmem_long_g_t shmem_long_g_handle = (shmem_long_g_t)get_function_handle("shmem_long_g");
  retval  =  shmem_long_g_handle ( a1,  a2);
  return retval;

}

extern long  shmem_long_g(long * a1, int a2) {
   __wrap_shmem_long_g(a1, a2);
}


/**********************************************************
   shmem_longlong_g
 **********************************************************/

extern long long  __wrap_shmem_longlong_g(long long * a1, int a2) ;
extern long long  __real_shmem_longlong_g(long long * a1, int a2)  {

  long long retval;
  typedef long long (*shmem_longlong_g_t)(long long * a1, int a2);
  shmem_longlong_g_t shmem_longlong_g_handle = (shmem_longlong_g_t)get_function_handle("shmem_longlong_g");
  retval  =  shmem_longlong_g_handle ( a1,  a2);
  return retval;

}

extern long long  shmem_longlong_g(long long * a1, int a2) {
   __wrap_shmem_longlong_g(a1, a2);
}


/**********************************************************
   shmem_float_g
 **********************************************************/

extern float  __wrap_shmem_float_g(float * a1, int a2) ;
extern float  __real_shmem_float_g(float * a1, int a2)  {

  float retval;
  typedef float (*shmem_float_g_t)(float * a1, int a2);
  shmem_float_g_t shmem_float_g_handle = (shmem_float_g_t)get_function_handle("shmem_float_g");
  retval  =  shmem_float_g_handle ( a1,  a2);
  return retval;

}

extern float  shmem_float_g(float * a1, int a2) {
   __wrap_shmem_float_g(a1, a2);
}


/**********************************************************
   shmem_double_g
 **********************************************************/

extern double  __wrap_shmem_double_g(double * a1, int a2) ;
extern double  __real_shmem_double_g(double * a1, int a2)  {

  double retval;
  typedef double (*shmem_double_g_t)(double * a1, int a2);
  shmem_double_g_t shmem_double_g_handle = (shmem_double_g_t)get_function_handle("shmem_double_g");
  retval  =  shmem_double_g_handle ( a1,  a2);
  return retval;

}

extern double  shmem_double_g(double * a1, int a2) {
   __wrap_shmem_double_g(a1, a2);
}


/**********************************************************
   shmem_longdouble_g
 **********************************************************/

extern long double  __wrap_shmem_longdouble_g(long double * a1, int a2) ;
extern long double  __real_shmem_longdouble_g(long double * a1, int a2)  {

  long double retval;
  typedef long double (*shmem_longdouble_g_t)(long double * a1, int a2);
  shmem_longdouble_g_t shmem_longdouble_g_handle = (shmem_longdouble_g_t)get_function_handle("shmem_longdouble_g");
  retval  =  shmem_longdouble_g_handle ( a1,  a2);
  return retval;

}

extern long double  shmem_longdouble_g(long double * a1, int a2) {
   __wrap_shmem_longdouble_g(a1, a2);
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


/**********************************************************
   shmem_iput32
 **********************************************************/

extern void  __wrap_shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iput32_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_iput32_t shmem_iput32_handle = (shmem_iput32_t)get_function_handle("shmem_iput32");
  shmem_iput32_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iput32(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iput64
 **********************************************************/

extern void  __wrap_shmem_iput64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iput64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iput64_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_iput64_t shmem_iput64_handle = (shmem_iput64_t)get_function_handle("shmem_iput64");
  shmem_iput64_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iput64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iput64(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iput128
 **********************************************************/

extern void  __wrap_shmem_iput128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iput128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iput128_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_iput128_t shmem_iput128_handle = (shmem_iput128_t)get_function_handle("shmem_iput128");
  shmem_iput128_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iput128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iput128(a1, a2, a3, a4, a5, a6);
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


/**********************************************************
   shmem_longdouble_iput
 **********************************************************/

extern void  __wrap_shmem_longdouble_iput(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_longdouble_iput(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_longdouble_iput_t)(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_longdouble_iput_t shmem_longdouble_iput_handle = (shmem_longdouble_iput_t)get_function_handle("shmem_longdouble_iput");
  shmem_longdouble_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longdouble_iput(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_longdouble_iput(a1, a2, a3, a4, a5, a6);
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


/**********************************************************
   shmem_iget32
 **********************************************************/

extern void  __wrap_shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iget32_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_iget32_t shmem_iget32_handle = (shmem_iget32_t)get_function_handle("shmem_iget32");
  shmem_iget32_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iget32(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iget64
 **********************************************************/

extern void  __wrap_shmem_iget64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iget64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iget64_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_iget64_t shmem_iget64_handle = (shmem_iget64_t)get_function_handle("shmem_iget64");
  shmem_iget64_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iget64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iget64(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iget128
 **********************************************************/

extern void  __wrap_shmem_iget128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iget128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iget128_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_iget128_t shmem_iget128_handle = (shmem_iget128_t)get_function_handle("shmem_iget128");
  shmem_iget128_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iget128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iget128(a1, a2, a3, a4, a5, a6);
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


/**********************************************************
   shmem_longdouble_iget
 **********************************************************/

extern void  __wrap_shmem_longdouble_iget(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_longdouble_iget(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_longdouble_iget_t)(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  shmem_longdouble_iget_t shmem_longdouble_iget_handle = (shmem_longdouble_iget_t)get_function_handle("shmem_longdouble_iget");
  shmem_longdouble_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longdouble_iget(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_longdouble_iget(a1, a2, a3, a4, a5, a6);
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


/**********************************************************
   shmem_addr_accessible
 **********************************************************/

extern int  __wrap_shmem_addr_accessible(const void * a1, int a2) ;
extern int  __real_shmem_addr_accessible(const void * a1, int a2)  {

  int retval;
  typedef int (*shmem_addr_accessible_t)(const void * a1, int a2);
  shmem_addr_accessible_t shmem_addr_accessible_handle = (shmem_addr_accessible_t)get_function_handle("shmem_addr_accessible");
  retval  =  shmem_addr_accessible_handle ( a1,  a2);
  return retval;

}

extern int  shmem_addr_accessible(const void * a1, int a2) {
   __wrap_shmem_addr_accessible(a1, a2);
}


/**********************************************************
   shmem_free
 **********************************************************/

extern void  __wrap_shmem_free(void * a1) ;
extern void  __real_shmem_free(void * a1)  {

  typedef void (*shmem_free_t)(void * a1);
  shmem_free_t shmem_free_handle = (shmem_free_t)get_function_handle("shmem_free");
  shmem_free_handle ( a1);

}

extern void  shmem_free(void * a1) {
   __wrap_shmem_free(a1);
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


/**********************************************************
   shmem_wait_until
 **********************************************************/

extern void  __wrap_shmem_wait_until(long * a1, int a2, long a3) ;
extern void  __real_shmem_wait_until(long * a1, int a2, long a3)  {

  typedef void (*shmem_wait_until_t)(long * a1, int a2, long a3);
  shmem_wait_until_t shmem_wait_until_handle = (shmem_wait_until_t)get_function_handle("shmem_wait_until");
  shmem_wait_until_handle ( a1,  a2,  a3);

}

extern void  shmem_wait_until(long * a1, int a2, long a3) {
   __wrap_shmem_wait_until(a1, a2, a3);
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


/**********************************************************
   shmem_wait
 **********************************************************/

extern void  __wrap_shmem_wait(long * a1, long a2) ;
extern void  __real_shmem_wait(long * a1, long a2)  {

  typedef void (*shmem_wait_t)(long * a1, long a2);
  shmem_wait_t shmem_wait_handle = (shmem_wait_t)get_function_handle("shmem_wait");
  shmem_wait_handle ( a1,  a2);

}

extern void  shmem_wait(long * a1, long a2) {
   __wrap_shmem_wait(a1, a2);
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


/**********************************************************
   shmem_swap
 **********************************************************/

extern long  __wrap_shmem_swap(long * a1, long a2, int a3) ;
extern long  __real_shmem_swap(long * a1, long a2, int a3)  {

  long retval;
  typedef long (*shmem_swap_t)(long * a1, long a2, int a3);
  shmem_swap_t shmem_swap_handle = (shmem_swap_t)get_function_handle("shmem_swap");
  retval  =  shmem_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern long  shmem_swap(long * a1, long a2, int a3) {
   __wrap_shmem_swap(a1, a2, a3);
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


/**********************************************************
   shmem_long_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_long_sum_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_sum_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_sum_to_all_t)(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_sum_to_all_t shmem_long_sum_to_all_handle = (shmem_long_sum_to_all_t)get_function_handle("shmem_long_sum_to_all");
  shmem_long_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_sum_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_complexd_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_complexd_sum_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8) ;
extern void  __real_shmem_complexd_sum_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8)  {

  typedef void (*shmem_complexd_sum_to_all_t)(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8);
  shmem_complexd_sum_to_all_t shmem_complexd_sum_to_all_handle = (shmem_complexd_sum_to_all_t)get_function_handle("shmem_complexd_sum_to_all");
  shmem_complexd_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_complexd_sum_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8) {
   __wrap_shmem_complexd_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_complexf_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_complexf_sum_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8) ;
extern void  __real_shmem_complexf_sum_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8)  {

  typedef void (*shmem_complexf_sum_to_all_t)(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8);
  shmem_complexf_sum_to_all_t shmem_complexf_sum_to_all_handle = (shmem_complexf_sum_to_all_t)get_function_handle("shmem_complexf_sum_to_all");
  shmem_complexf_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_complexf_sum_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8) {
   __wrap_shmem_complexf_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_double_sum_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_sum_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_sum_to_all_t)(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8);
  shmem_double_sum_to_all_t shmem_double_sum_to_all_handle = (shmem_double_sum_to_all_t)get_function_handle("shmem_double_sum_to_all");
  shmem_double_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_sum_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_float_sum_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_sum_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_sum_to_all_t)(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8);
  shmem_float_sum_to_all_t shmem_float_sum_to_all_handle = (shmem_float_sum_to_all_t)get_function_handle("shmem_float_sum_to_all");
  shmem_float_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_sum_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_int_sum_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_sum_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_sum_to_all_t)(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_sum_to_all_t shmem_int_sum_to_all_handle = (shmem_int_sum_to_all_t)get_function_handle("shmem_int_sum_to_all");
  shmem_int_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_sum_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longdouble_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_longdouble_sum_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_longdouble_sum_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_longdouble_sum_to_all_t)(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8);
  shmem_longdouble_sum_to_all_t shmem_longdouble_sum_to_all_handle = (shmem_longdouble_sum_to_all_t)get_function_handle("shmem_longdouble_sum_to_all");
  shmem_longdouble_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longdouble_sum_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_longdouble_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_sum_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_sum_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_sum_to_all_t)(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_sum_to_all_t shmem_longlong_sum_to_all_handle = (shmem_longlong_sum_to_all_t)get_function_handle("shmem_longlong_sum_to_all");
  shmem_longlong_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_sum_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_short_sum_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_sum_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_sum_to_all_t)(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_sum_to_all_t shmem_short_sum_to_all_handle = (shmem_short_sum_to_all_t)get_function_handle("shmem_short_sum_to_all");
  shmem_short_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_sum_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_complexd_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_complexd_prod_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8) ;
extern void  __real_shmem_complexd_prod_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8)  {

  typedef void (*shmem_complexd_prod_to_all_t)(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8);
  shmem_complexd_prod_to_all_t shmem_complexd_prod_to_all_handle = (shmem_complexd_prod_to_all_t)get_function_handle("shmem_complexd_prod_to_all");
  shmem_complexd_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_complexd_prod_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8) {
   __wrap_shmem_complexd_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_complexf_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_complexf_prod_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8) ;
extern void  __real_shmem_complexf_prod_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8)  {

  typedef void (*shmem_complexf_prod_to_all_t)(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8);
  shmem_complexf_prod_to_all_t shmem_complexf_prod_to_all_handle = (shmem_complexf_prod_to_all_t)get_function_handle("shmem_complexf_prod_to_all");
  shmem_complexf_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_complexf_prod_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8) {
   __wrap_shmem_complexf_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_double_prod_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_prod_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_prod_to_all_t)(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8);
  shmem_double_prod_to_all_t shmem_double_prod_to_all_handle = (shmem_double_prod_to_all_t)get_function_handle("shmem_double_prod_to_all");
  shmem_double_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_prod_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_float_prod_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_prod_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_prod_to_all_t)(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8);
  shmem_float_prod_to_all_t shmem_float_prod_to_all_handle = (shmem_float_prod_to_all_t)get_function_handle("shmem_float_prod_to_all");
  shmem_float_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_prod_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_int_prod_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_prod_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_prod_to_all_t)(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_prod_to_all_t shmem_int_prod_to_all_handle = (shmem_int_prod_to_all_t)get_function_handle("shmem_int_prod_to_all");
  shmem_int_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_prod_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_long_prod_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_prod_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_prod_to_all_t)(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_prod_to_all_t shmem_long_prod_to_all_handle = (shmem_long_prod_to_all_t)get_function_handle("shmem_long_prod_to_all");
  shmem_long_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_prod_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longdouble_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_longdouble_prod_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_longdouble_prod_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_longdouble_prod_to_all_t)(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8);
  shmem_longdouble_prod_to_all_t shmem_longdouble_prod_to_all_handle = (shmem_longdouble_prod_to_all_t)get_function_handle("shmem_longdouble_prod_to_all");
  shmem_longdouble_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longdouble_prod_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_longdouble_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_prod_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_prod_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_prod_to_all_t)(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_prod_to_all_t shmem_longlong_prod_to_all_handle = (shmem_longlong_prod_to_all_t)get_function_handle("shmem_longlong_prod_to_all");
  shmem_longlong_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_prod_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_short_prod_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_prod_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_prod_to_all_t)(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_prod_to_all_t shmem_short_prod_to_all_handle = (shmem_short_prod_to_all_t)get_function_handle("shmem_short_prod_to_all");
  shmem_short_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_prod_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_and_to_all
 **********************************************************/

extern void  __wrap_shmem_int_and_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_and_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_and_to_all_t)(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_and_to_all_t shmem_int_and_to_all_handle = (shmem_int_and_to_all_t)get_function_handle("shmem_int_and_to_all");
  shmem_int_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_and_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_and_to_all
 **********************************************************/

extern void  __wrap_shmem_long_and_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_and_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_and_to_all_t)(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_and_to_all_t shmem_long_and_to_all_handle = (shmem_long_and_to_all_t)get_function_handle("shmem_long_and_to_all");
  shmem_long_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_and_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_and_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_and_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_and_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_and_to_all_t)(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_and_to_all_t shmem_longlong_and_to_all_handle = (shmem_longlong_and_to_all_t)get_function_handle("shmem_longlong_and_to_all");
  shmem_longlong_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_and_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_and_to_all
 **********************************************************/

extern void  __wrap_shmem_short_and_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_and_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_and_to_all_t)(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_and_to_all_t shmem_short_and_to_all_handle = (shmem_short_and_to_all_t)get_function_handle("shmem_short_and_to_all");
  shmem_short_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_and_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_or_to_all
 **********************************************************/

extern void  __wrap_shmem_int_or_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_or_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_or_to_all_t)(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_or_to_all_t shmem_int_or_to_all_handle = (shmem_int_or_to_all_t)get_function_handle("shmem_int_or_to_all");
  shmem_int_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_or_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_or_to_all
 **********************************************************/

extern void  __wrap_shmem_long_or_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_or_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_or_to_all_t)(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_or_to_all_t shmem_long_or_to_all_handle = (shmem_long_or_to_all_t)get_function_handle("shmem_long_or_to_all");
  shmem_long_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_or_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_or_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_or_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_or_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_or_to_all_t)(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_or_to_all_t shmem_longlong_or_to_all_handle = (shmem_longlong_or_to_all_t)get_function_handle("shmem_longlong_or_to_all");
  shmem_longlong_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_or_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_or_to_all
 **********************************************************/

extern void  __wrap_shmem_short_or_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_or_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_or_to_all_t)(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_or_to_all_t shmem_short_or_to_all_handle = (shmem_short_or_to_all_t)get_function_handle("shmem_short_or_to_all");
  shmem_short_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_or_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_int_xor_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_xor_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_xor_to_all_t)(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_xor_to_all_t shmem_int_xor_to_all_handle = (shmem_int_xor_to_all_t)get_function_handle("shmem_int_xor_to_all");
  shmem_int_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_xor_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_long_xor_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_xor_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_xor_to_all_t)(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_xor_to_all_t shmem_long_xor_to_all_handle = (shmem_long_xor_to_all_t)get_function_handle("shmem_long_xor_to_all");
  shmem_long_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_xor_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_xor_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_xor_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_xor_to_all_t)(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_xor_to_all_t shmem_longlong_xor_to_all_handle = (shmem_longlong_xor_to_all_t)get_function_handle("shmem_longlong_xor_to_all");
  shmem_longlong_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_xor_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_short_xor_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_xor_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_xor_to_all_t)(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_xor_to_all_t shmem_short_xor_to_all_handle = (shmem_short_xor_to_all_t)get_function_handle("shmem_short_xor_to_all");
  shmem_short_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_xor_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_max_to_all
 **********************************************************/

extern void  __wrap_shmem_int_max_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_max_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_max_to_all_t)(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_max_to_all_t shmem_int_max_to_all_handle = (shmem_int_max_to_all_t)get_function_handle("shmem_int_max_to_all");
  shmem_int_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_max_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_max_to_all
 **********************************************************/

extern void  __wrap_shmem_long_max_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_max_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_max_to_all_t)(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_max_to_all_t shmem_long_max_to_all_handle = (shmem_long_max_to_all_t)get_function_handle("shmem_long_max_to_all");
  shmem_long_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_max_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_max_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_max_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_max_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_max_to_all_t)(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_max_to_all_t shmem_longlong_max_to_all_handle = (shmem_longlong_max_to_all_t)get_function_handle("shmem_longlong_max_to_all");
  shmem_longlong_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_max_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_max_to_all
 **********************************************************/

extern void  __wrap_shmem_short_max_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_max_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_max_to_all_t)(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_max_to_all_t shmem_short_max_to_all_handle = (shmem_short_max_to_all_t)get_function_handle("shmem_short_max_to_all");
  shmem_short_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_max_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longdouble_max_to_all
 **********************************************************/

extern void  __wrap_shmem_longdouble_max_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_longdouble_max_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_longdouble_max_to_all_t)(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8);
  shmem_longdouble_max_to_all_t shmem_longdouble_max_to_all_handle = (shmem_longdouble_max_to_all_t)get_function_handle("shmem_longdouble_max_to_all");
  shmem_longdouble_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longdouble_max_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_longdouble_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_max_to_all
 **********************************************************/

extern void  __wrap_shmem_float_max_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_max_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_max_to_all_t)(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8);
  shmem_float_max_to_all_t shmem_float_max_to_all_handle = (shmem_float_max_to_all_t)get_function_handle("shmem_float_max_to_all");
  shmem_float_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_max_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_max_to_all
 **********************************************************/

extern void  __wrap_shmem_double_max_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_max_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_max_to_all_t)(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8);
  shmem_double_max_to_all_t shmem_double_max_to_all_handle = (shmem_double_max_to_all_t)get_function_handle("shmem_double_max_to_all");
  shmem_double_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_max_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_min_to_all
 **********************************************************/

extern void  __wrap_shmem_int_min_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_min_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_min_to_all_t)(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8);
  shmem_int_min_to_all_t shmem_int_min_to_all_handle = (shmem_int_min_to_all_t)get_function_handle("shmem_int_min_to_all");
  shmem_int_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_min_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_min_to_all
 **********************************************************/

extern void  __wrap_shmem_long_min_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_min_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_min_to_all_t)(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8);
  shmem_long_min_to_all_t shmem_long_min_to_all_handle = (shmem_long_min_to_all_t)get_function_handle("shmem_long_min_to_all");
  shmem_long_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_min_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_min_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_min_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_min_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_min_to_all_t)(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8);
  shmem_longlong_min_to_all_t shmem_longlong_min_to_all_handle = (shmem_longlong_min_to_all_t)get_function_handle("shmem_longlong_min_to_all");
  shmem_longlong_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_min_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_min_to_all
 **********************************************************/

extern void  __wrap_shmem_short_min_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_min_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_min_to_all_t)(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8);
  shmem_short_min_to_all_t shmem_short_min_to_all_handle = (shmem_short_min_to_all_t)get_function_handle("shmem_short_min_to_all");
  shmem_short_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_min_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longdouble_min_to_all
 **********************************************************/

extern void  __wrap_shmem_longdouble_min_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_longdouble_min_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_longdouble_min_to_all_t)(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8);
  shmem_longdouble_min_to_all_t shmem_longdouble_min_to_all_handle = (shmem_longdouble_min_to_all_t)get_function_handle("shmem_longdouble_min_to_all");
  shmem_longdouble_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longdouble_min_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_longdouble_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_min_to_all
 **********************************************************/

extern void  __wrap_shmem_float_min_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_min_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_min_to_all_t)(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8);
  shmem_float_min_to_all_t shmem_float_min_to_all_handle = (shmem_float_min_to_all_t)get_function_handle("shmem_float_min_to_all");
  shmem_float_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_min_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_min_to_all
 **********************************************************/

extern void  __wrap_shmem_double_min_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_min_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_min_to_all_t)(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8);
  shmem_double_min_to_all_t shmem_double_min_to_all_handle = (shmem_double_min_to_all_t)get_function_handle("shmem_double_min_to_all");
  shmem_double_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_min_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_broadcast64
 **********************************************************/

extern void  __wrap_shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) ;
extern void  __real_shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8)  {

  typedef void (*shmem_broadcast64_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8);
  shmem_broadcast64_t shmem_broadcast64_handle = (shmem_broadcast64_t)get_function_handle("shmem_broadcast64");
  shmem_broadcast64_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) {
   __wrap_shmem_broadcast64(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_broadcast32
 **********************************************************/

extern void  __wrap_shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) ;
extern void  __real_shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8)  {

  typedef void (*shmem_broadcast32_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8);
  shmem_broadcast32_t shmem_broadcast32_handle = (shmem_broadcast32_t)get_function_handle("shmem_broadcast32");
  shmem_broadcast32_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) {
   __wrap_shmem_broadcast32(a1, a2, a3, a4, a5, a6, a7, a8);
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


/**********************************************************
   shmemx_short_put_nb
 **********************************************************/

extern void  __wrap_shmemx_short_put_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_short_put_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_short_put_nb_t)(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_short_put_nb_t shmemx_short_put_nb_handle = (shmemx_short_put_nb_t)get_function_handle("shmemx_short_put_nb");
  shmemx_short_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_short_put_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_short_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_int_put_nb
 **********************************************************/

extern void  __wrap_shmemx_int_put_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_int_put_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_int_put_nb_t)(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_int_put_nb_t shmemx_int_put_nb_handle = (shmemx_int_put_nb_t)get_function_handle("shmemx_int_put_nb");
  shmemx_int_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_int_put_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_int_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_long_put_nb
 **********************************************************/

extern void  __wrap_shmemx_long_put_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_long_put_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_long_put_nb_t)(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_long_put_nb_t shmemx_long_put_nb_handle = (shmemx_long_put_nb_t)get_function_handle("shmemx_long_put_nb");
  shmemx_long_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_long_put_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_long_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_longlong_put_nb
 **********************************************************/

extern void  __wrap_shmemx_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_longlong_put_nb_t)(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_longlong_put_nb_t shmemx_longlong_put_nb_handle = (shmemx_longlong_put_nb_t)get_function_handle("shmemx_longlong_put_nb");
  shmemx_longlong_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_longlong_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_longdouble_put_nb
 **********************************************************/

extern void  __wrap_shmemx_longdouble_put_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_longdouble_put_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_longdouble_put_nb_t)(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_longdouble_put_nb_t shmemx_longdouble_put_nb_handle = (shmemx_longdouble_put_nb_t)get_function_handle("shmemx_longdouble_put_nb");
  shmemx_longdouble_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_longdouble_put_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_longdouble_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_double_put_nb
 **********************************************************/

extern void  __wrap_shmemx_double_put_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_double_put_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_double_put_nb_t)(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_double_put_nb_t shmemx_double_put_nb_handle = (shmemx_double_put_nb_t)get_function_handle("shmemx_double_put_nb");
  shmemx_double_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_double_put_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_double_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_float_put_nb
 **********************************************************/

extern void  __wrap_shmemx_float_put_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_float_put_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_float_put_nb_t)(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_float_put_nb_t shmemx_float_put_nb_handle = (shmemx_float_put_nb_t)get_function_handle("shmemx_float_put_nb");
  shmemx_float_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_float_put_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_float_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_putmem_nb
 **********************************************************/

extern void  __wrap_shmemx_putmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_putmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_putmem_nb_t)(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_putmem_nb_t shmemx_putmem_nb_handle = (shmemx_putmem_nb_t)get_function_handle("shmemx_putmem_nb");
  shmemx_putmem_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_putmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_putmem_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_put32_nb
 **********************************************************/

extern void  __wrap_shmemx_put32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_put32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_put32_nb_t)(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_put32_nb_t shmemx_put32_nb_handle = (shmemx_put32_nb_t)get_function_handle("shmemx_put32_nb");
  shmemx_put32_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_put32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_put32_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_put64_nb
 **********************************************************/

extern void  __wrap_shmemx_put64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_put64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_put64_nb_t)(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_put64_nb_t shmemx_put64_nb_handle = (shmemx_put64_nb_t)get_function_handle("shmemx_put64_nb");
  shmemx_put64_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_put64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_put64_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_put128_nb
 **********************************************************/

extern void  __wrap_shmemx_put128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_put128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_put128_nb_t)(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_put128_nb_t shmemx_put128_nb_handle = (shmemx_put128_nb_t)get_function_handle("shmemx_put128_nb");
  shmemx_put128_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_put128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_put128_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_short_get_nb
 **********************************************************/

extern void  __wrap_shmemx_short_get_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_short_get_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_short_get_nb_t)(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_short_get_nb_t shmemx_short_get_nb_handle = (shmemx_short_get_nb_t)get_function_handle("shmemx_short_get_nb");
  shmemx_short_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_short_get_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_short_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_int_get_nb
 **********************************************************/

extern void  __wrap_shmemx_int_get_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_int_get_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_int_get_nb_t)(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_int_get_nb_t shmemx_int_get_nb_handle = (shmemx_int_get_nb_t)get_function_handle("shmemx_int_get_nb");
  shmemx_int_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_int_get_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_int_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_long_get_nb
 **********************************************************/

extern void  __wrap_shmemx_long_get_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_long_get_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_long_get_nb_t)(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_long_get_nb_t shmemx_long_get_nb_handle = (shmemx_long_get_nb_t)get_function_handle("shmemx_long_get_nb");
  shmemx_long_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_long_get_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_long_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_longlong_get_nb
 **********************************************************/

extern void  __wrap_shmemx_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_longlong_get_nb_t)(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_longlong_get_nb_t shmemx_longlong_get_nb_handle = (shmemx_longlong_get_nb_t)get_function_handle("shmemx_longlong_get_nb");
  shmemx_longlong_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_longlong_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_longdouble_get_nb
 **********************************************************/

extern void  __wrap_shmemx_longdouble_get_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_longdouble_get_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_longdouble_get_nb_t)(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_longdouble_get_nb_t shmemx_longdouble_get_nb_handle = (shmemx_longdouble_get_nb_t)get_function_handle("shmemx_longdouble_get_nb");
  shmemx_longdouble_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_longdouble_get_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_longdouble_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_double_get_nb
 **********************************************************/

extern void  __wrap_shmemx_double_get_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_double_get_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_double_get_nb_t)(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_double_get_nb_t shmemx_double_get_nb_handle = (shmemx_double_get_nb_t)get_function_handle("shmemx_double_get_nb");
  shmemx_double_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_double_get_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_double_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_float_get_nb
 **********************************************************/

extern void  __wrap_shmemx_float_get_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_float_get_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_float_get_nb_t)(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_float_get_nb_t shmemx_float_get_nb_handle = (shmemx_float_get_nb_t)get_function_handle("shmemx_float_get_nb");
  shmemx_float_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_float_get_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_float_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_getmem_nb
 **********************************************************/

extern void  __wrap_shmemx_getmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_getmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_getmem_nb_t)(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_getmem_nb_t shmemx_getmem_nb_handle = (shmemx_getmem_nb_t)get_function_handle("shmemx_getmem_nb");
  shmemx_getmem_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_getmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_getmem_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_get32_nb
 **********************************************************/

extern void  __wrap_shmemx_get32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_get32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_get32_nb_t)(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_get32_nb_t shmemx_get32_nb_handle = (shmemx_get32_nb_t)get_function_handle("shmemx_get32_nb");
  shmemx_get32_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_get32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_get32_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_get64_nb
 **********************************************************/

extern void  __wrap_shmemx_get64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_get64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_get64_nb_t)(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_get64_nb_t shmemx_get64_nb_handle = (shmemx_get64_nb_t)get_function_handle("shmemx_get64_nb");
  shmemx_get64_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_get64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_get64_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_get128_nb
 **********************************************************/

extern void  __wrap_shmemx_get128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __real_shmemx_get128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  typedef void (*shmemx_get128_nb_t)(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5);
  shmemx_get128_nb_t shmemx_get128_nb_handle = (shmemx_get128_nb_t)get_function_handle("shmemx_get128_nb");
  shmemx_get128_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmemx_get128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) {
   __wrap_shmemx_get128_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmemx_wait_req
 **********************************************************/

extern void  __wrap_shmemx_wait_req(shmemx_request_handle_t a1) ;
extern void  __real_shmemx_wait_req(shmemx_request_handle_t a1)  {

  typedef void (*shmemx_wait_req_t)(shmemx_request_handle_t a1);
  shmemx_wait_req_t shmemx_wait_req_handle = (shmemx_wait_req_t)get_function_handle("shmemx_wait_req");
  shmemx_wait_req_handle ( a1);

}

extern void  shmemx_wait_req(shmemx_request_handle_t a1) {
   __wrap_shmemx_wait_req(a1);
}


/**********************************************************
   shmemx_test_req
 **********************************************************/

extern void  __wrap_shmemx_test_req(shmemx_request_handle_t a1, int * a2) ;
extern void  __real_shmemx_test_req(shmemx_request_handle_t a1, int * a2)  {

  typedef void (*shmemx_test_req_t)(shmemx_request_handle_t a1, int * a2);
  shmemx_test_req_t shmemx_test_req_handle = (shmemx_test_req_t)get_function_handle("shmemx_test_req");
  shmemx_test_req_handle ( a1,  a2);

}

extern void  shmemx_test_req(shmemx_request_handle_t a1, int * a2) {
   __wrap_shmemx_test_req(a1, a2);
}


/**********************************************************
   shmemx_int_xor
 **********************************************************/

extern void  __wrap_shmemx_int_xor(int * a1, int a2, int a3) ;
extern void  __real_shmemx_int_xor(int * a1, int a2, int a3)  {

  typedef void (*shmemx_int_xor_t)(int * a1, int a2, int a3);
  shmemx_int_xor_t shmemx_int_xor_handle = (shmemx_int_xor_t)get_function_handle("shmemx_int_xor");
  shmemx_int_xor_handle ( a1,  a2,  a3);

}

extern void  shmemx_int_xor(int * a1, int a2, int a3) {
   __wrap_shmemx_int_xor(a1, a2, a3);
}


/**********************************************************
   shmemx_long_xor
 **********************************************************/

extern void  __wrap_shmemx_long_xor(long * a1, long a2, int a3) ;
extern void  __real_shmemx_long_xor(long * a1, long a2, int a3)  {

  typedef void (*shmemx_long_xor_t)(long * a1, long a2, int a3);
  shmemx_long_xor_t shmemx_long_xor_handle = (shmemx_long_xor_t)get_function_handle("shmemx_long_xor");
  shmemx_long_xor_handle ( a1,  a2,  a3);

}

extern void  shmemx_long_xor(long * a1, long a2, int a3) {
   __wrap_shmemx_long_xor(a1, a2, a3);
}


/**********************************************************
   shmemx_longlong_xor
 **********************************************************/

extern void  __wrap_shmemx_longlong_xor(long long * a1, long long a2, int a3) ;
extern void  __real_shmemx_longlong_xor(long long * a1, long long a2, int a3)  {

  typedef void (*shmemx_longlong_xor_t)(long long * a1, long long a2, int a3);
  shmemx_longlong_xor_t shmemx_longlong_xor_handle = (shmemx_longlong_xor_t)get_function_handle("shmemx_longlong_xor");
  shmemx_longlong_xor_handle ( a1,  a2,  a3);

}

extern void  shmemx_longlong_xor(long long * a1, long long a2, int a3) {
   __wrap_shmemx_longlong_xor(a1, a2, a3);
}


/**********************************************************
   shmemx_int_fetch
 **********************************************************/

extern int  __wrap_shmemx_int_fetch(int * a1, int a2) ;
extern int  __real_shmemx_int_fetch(int * a1, int a2)  {

  int retval;
  typedef int (*shmemx_int_fetch_t)(int * a1, int a2);
  shmemx_int_fetch_t shmemx_int_fetch_handle = (shmemx_int_fetch_t)get_function_handle("shmemx_int_fetch");
  retval  =  shmemx_int_fetch_handle ( a1,  a2);
  return retval;

}

extern int  shmemx_int_fetch(int * a1, int a2) {
   __wrap_shmemx_int_fetch(a1, a2);
}


/**********************************************************
   shmemx_long_fetch
 **********************************************************/

extern long  __wrap_shmemx_long_fetch(long * a1, int a2) ;
extern long  __real_shmemx_long_fetch(long * a1, int a2)  {

  long retval;
  typedef long (*shmemx_long_fetch_t)(long * a1, int a2);
  shmemx_long_fetch_t shmemx_long_fetch_handle = (shmemx_long_fetch_t)get_function_handle("shmemx_long_fetch");
  retval  =  shmemx_long_fetch_handle ( a1,  a2);
  return retval;

}

extern long  shmemx_long_fetch(long * a1, int a2) {
   __wrap_shmemx_long_fetch(a1, a2);
}


/**********************************************************
   shmemx_longlong_fetch
 **********************************************************/

extern long long  __wrap_shmemx_longlong_fetch(long long * a1, int a2) ;
extern long long  __real_shmemx_longlong_fetch(long long * a1, int a2)  {

  long long retval;
  typedef long long (*shmemx_longlong_fetch_t)(long long * a1, int a2);
  shmemx_longlong_fetch_t shmemx_longlong_fetch_handle = (shmemx_longlong_fetch_t)get_function_handle("shmemx_longlong_fetch");
  retval  =  shmemx_longlong_fetch_handle ( a1,  a2);
  return retval;

}

extern long long  shmemx_longlong_fetch(long long * a1, int a2) {
   __wrap_shmemx_longlong_fetch(a1, a2);
}


/**********************************************************
   shmemx_int_set
 **********************************************************/

extern void  __wrap_shmemx_int_set(int * a1, int a2, int a3) ;
extern void  __real_shmemx_int_set(int * a1, int a2, int a3)  {

  typedef void (*shmemx_int_set_t)(int * a1, int a2, int a3);
  shmemx_int_set_t shmemx_int_set_handle = (shmemx_int_set_t)get_function_handle("shmemx_int_set");
  shmemx_int_set_handle ( a1,  a2,  a3);

}

extern void  shmemx_int_set(int * a1, int a2, int a3) {
   __wrap_shmemx_int_set(a1, a2, a3);
}


/**********************************************************
   shmemx_long_set
 **********************************************************/

extern void  __wrap_shmemx_long_set(long * a1, long a2, int a3) ;
extern void  __real_shmemx_long_set(long * a1, long a2, int a3)  {

  typedef void (*shmemx_long_set_t)(long * a1, long a2, int a3);
  shmemx_long_set_t shmemx_long_set_handle = (shmemx_long_set_t)get_function_handle("shmemx_long_set");
  shmemx_long_set_handle ( a1,  a2,  a3);

}

extern void  shmemx_long_set(long * a1, long a2, int a3) {
   __wrap_shmemx_long_set(a1, a2, a3);
}


/**********************************************************
   shmemx_longlong_set
 **********************************************************/

extern void  __wrap_shmemx_longlong_set(long long * a1, long long a2, int a3) ;
extern void  __real_shmemx_longlong_set(long long * a1, long long a2, int a3)  {

  typedef void (*shmemx_longlong_set_t)(long long * a1, long long a2, int a3);
  shmemx_longlong_set_t shmemx_longlong_set_handle = (shmemx_longlong_set_t)get_function_handle("shmemx_longlong_set");
  shmemx_longlong_set_handle ( a1,  a2,  a3);

}

extern void  shmemx_longlong_set(long long * a1, long long a2, int a3) {
   __wrap_shmemx_longlong_set(a1, a2, a3);
}


/**********************************************************
   shmemx_wtime
 **********************************************************/

extern double  __wrap_shmemx_wtime() ;
extern double  __real_shmemx_wtime()  {

  double retval;
  typedef double (*shmemx_wtime_t)();
  shmemx_wtime_t shmemx_wtime_handle = (shmemx_wtime_t)get_function_handle("shmemx_wtime");
  retval  =  shmemx_wtime_handle ();
  return retval;

}

extern double  shmemx_wtime() {
   __wrap_shmemx_wtime();
}


/**********************************************************
   shmemx_fence_test
 **********************************************************/

extern int  __wrap_shmemx_fence_test() ;
extern int  __real_shmemx_fence_test()  {

  int retval;
  typedef int (*shmemx_fence_test_t)();
  shmemx_fence_test_t shmemx_fence_test_handle = (shmemx_fence_test_t)get_function_handle("shmemx_fence_test");
  retval  =  shmemx_fence_test_handle ();
  return retval;

}

extern int  shmemx_fence_test() {
   __wrap_shmemx_fence_test();
}


/**********************************************************
   shmemx_quiet_test
 **********************************************************/

extern int  __wrap_shmemx_quiet_test() ;
extern int  __real_shmemx_quiet_test()  {

  int retval;
  typedef int (*shmemx_quiet_test_t)();
  shmemx_quiet_test_t shmemx_quiet_test_handle = (shmemx_quiet_test_t)get_function_handle("shmemx_quiet_test");
  retval  =  shmemx_quiet_test_handle ();
  return retval;

}

extern int  shmemx_quiet_test() {
   __wrap_shmemx_quiet_test();
}

