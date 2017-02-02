#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <tau_shmem.h>
#include <Profile/Profiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>


static void * get_function_handle(char const * name)
{
  char const * err;
  void * handle;

  // Reset error pointer
  dlerror();

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
   shmem_get8
 **********************************************************/

extern void  __wrap_shmem_get8(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get8(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get8_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_get8_t shmem_get8_handle = (shmem_get8_t)NULL;
  if (!shmem_get8_handle) {
    shmem_get8_handle = get_function_handle("shmem_get8");
  }

  shmem_get8_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get8(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get8(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get16
 **********************************************************/

extern void  __wrap_shmem_get16(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get16(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get16_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_get16_t shmem_get16_handle = (shmem_get16_t)NULL;
  if (!shmem_get16_handle) {
    shmem_get16_handle = get_function_handle("shmem_get16");
  }

  shmem_get16_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get16(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get16(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get32
 **********************************************************/

extern void  __wrap_shmem_get32(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get32(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get32_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_get32_t shmem_get32_handle = (shmem_get32_t)NULL;
  if (!shmem_get32_handle) {
    shmem_get32_handle = get_function_handle("shmem_get32");
  }

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
  static shmem_get64_t shmem_get64_handle = (shmem_get64_t)NULL;
  if (!shmem_get64_handle) {
    shmem_get64_handle = get_function_handle("shmem_get64");
  }

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
  static shmem_get128_t shmem_get128_handle = (shmem_get128_t)NULL;
  if (!shmem_get128_handle) {
    shmem_get128_handle = get_function_handle("shmem_get128");
  }

  shmem_get128_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get128(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get128(a1, a2, a3, a4);
}


/**********************************************************
   shmem_getmem
 **********************************************************/

extern void  __wrap_shmem_getmem(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_getmem(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_getmem_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_getmem_t shmem_getmem_handle = (shmem_getmem_t)NULL;
  if (!shmem_getmem_handle) {
    shmem_getmem_handle = get_function_handle("shmem_getmem");
  }

  shmem_getmem_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_getmem(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_getmem(a1, a2, a3, a4);
}


/**********************************************************
   shmem_char_get
 **********************************************************/

extern void  __wrap_shmem_char_get(char * a1, const char * a2, size_t a3, int a4) ;
extern void  __real_shmem_char_get(char * a1, const char * a2, size_t a3, int a4)  {

  typedef void (*shmem_char_get_t)(char * a1, const char * a2, size_t a3, int a4);
  static shmem_char_get_t shmem_char_get_handle = (shmem_char_get_t)NULL;
  if (!shmem_char_get_handle) {
    shmem_char_get_handle = get_function_handle("shmem_char_get");
  }

  shmem_char_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_char_get(char * a1, const char * a2, size_t a3, int a4) {
   __wrap_shmem_char_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_get
 **********************************************************/

extern void  __wrap_shmem_short_get(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __real_shmem_short_get(short * a1, const short * a2, size_t a3, int a4)  {

  typedef void (*shmem_short_get_t)(short * a1, const short * a2, size_t a3, int a4);
  static shmem_short_get_t shmem_short_get_handle = (shmem_short_get_t)NULL;
  if (!shmem_short_get_handle) {
    shmem_short_get_handle = get_function_handle("shmem_short_get");
  }

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
  static shmem_int_get_t shmem_int_get_handle = (shmem_int_get_t)NULL;
  if (!shmem_int_get_handle) {
    shmem_int_get_handle = get_function_handle("shmem_int_get");
  }

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
  static shmem_long_get_t shmem_long_get_handle = (shmem_long_get_t)NULL;
  if (!shmem_long_get_handle) {
    shmem_long_get_handle = get_function_handle("shmem_long_get");
  }

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
  static shmem_longlong_get_t shmem_longlong_get_handle = (shmem_longlong_get_t)NULL;
  if (!shmem_longlong_get_handle) {
    shmem_longlong_get_handle = get_function_handle("shmem_longlong_get");
  }

  shmem_longlong_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_get(long long * a1, const long long * a2, size_t a3, int a4) {
   __wrap_shmem_longlong_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float_get
 **********************************************************/

extern void  __wrap_shmem_float_get(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __real_shmem_float_get(float * a1, const float * a2, size_t a3, int a4)  {

  typedef void (*shmem_float_get_t)(float * a1, const float * a2, size_t a3, int a4);
  static shmem_float_get_t shmem_float_get_handle = (shmem_float_get_t)NULL;
  if (!shmem_float_get_handle) {
    shmem_float_get_handle = get_function_handle("shmem_float_get");
  }

  shmem_float_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float_get(float * a1, const float * a2, size_t a3, int a4) {
   __wrap_shmem_float_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_double_get
 **********************************************************/

extern void  __wrap_shmem_double_get(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __real_shmem_double_get(double * a1, const double * a2, size_t a3, int a4)  {

  typedef void (*shmem_double_get_t)(double * a1, const double * a2, size_t a3, int a4);
  static shmem_double_get_t shmem_double_get_handle = (shmem_double_get_t)NULL;
  if (!shmem_double_get_handle) {
    shmem_double_get_handle = get_function_handle("shmem_double_get");
  }

  shmem_double_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_double_get(double * a1, const double * a2, size_t a3, int a4) {
   __wrap_shmem_double_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float128_get
 **********************************************************/

extern void  __wrap_shmem_float128_get(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __real_shmem_float128_get(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  typedef void (*shmem_float128_get_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4);
  static shmem_float128_get_t shmem_float128_get_handle = (shmem_float128_get_t)NULL;
  if (!shmem_float128_get_handle) {
    shmem_float128_get_handle = get_function_handle("shmem_float128_get");
  }

  shmem_float128_get_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float128_get(__float128 * a1, const __float128 * a2, size_t a3, int a4) {
   __wrap_shmem_float128_get(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put8
 **********************************************************/

extern void  __wrap_shmem_put8(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put8(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put8_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_put8_t shmem_put8_handle = (shmem_put8_t)NULL;
  if (!shmem_put8_handle) {
    shmem_put8_handle = get_function_handle("shmem_put8");
  }

  shmem_put8_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put8(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put8(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put16
 **********************************************************/

extern void  __wrap_shmem_put16(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put16(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put16_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_put16_t shmem_put16_handle = (shmem_put16_t)NULL;
  if (!shmem_put16_handle) {
    shmem_put16_handle = get_function_handle("shmem_put16");
  }

  shmem_put16_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put16(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put16(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put32
 **********************************************************/

extern void  __wrap_shmem_put32(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put32(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put32_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_put32_t shmem_put32_handle = (shmem_put32_t)NULL;
  if (!shmem_put32_handle) {
    shmem_put32_handle = get_function_handle("shmem_put32");
  }

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
  static shmem_put64_t shmem_put64_handle = (shmem_put64_t)NULL;
  if (!shmem_put64_handle) {
    shmem_put64_handle = get_function_handle("shmem_put64");
  }

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
  static shmem_put128_t shmem_put128_handle = (shmem_put128_t)NULL;
  if (!shmem_put128_handle) {
    shmem_put128_handle = get_function_handle("shmem_put128");
  }

  shmem_put128_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put128(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put128(a1, a2, a3, a4);
}


/**********************************************************
   shmem_putmem
 **********************************************************/

extern void  __wrap_shmem_putmem(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_putmem(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_putmem_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_putmem_t shmem_putmem_handle = (shmem_putmem_t)NULL;
  if (!shmem_putmem_handle) {
    shmem_putmem_handle = get_function_handle("shmem_putmem");
  }

  shmem_putmem_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_putmem(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_putmem(a1, a2, a3, a4);
}


/**********************************************************
   shmem_char_put
 **********************************************************/

extern void  __wrap_shmem_char_put(char * a1, const char * a2, size_t a3, int a4) ;
extern void  __real_shmem_char_put(char * a1, const char * a2, size_t a3, int a4)  {

  typedef void (*shmem_char_put_t)(char * a1, const char * a2, size_t a3, int a4);
  static shmem_char_put_t shmem_char_put_handle = (shmem_char_put_t)NULL;
  if (!shmem_char_put_handle) {
    shmem_char_put_handle = get_function_handle("shmem_char_put");
  }

  shmem_char_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_char_put(char * a1, const char * a2, size_t a3, int a4) {
   __wrap_shmem_char_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_put
 **********************************************************/

extern void  __wrap_shmem_short_put(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __real_shmem_short_put(short * a1, const short * a2, size_t a3, int a4)  {

  typedef void (*shmem_short_put_t)(short * a1, const short * a2, size_t a3, int a4);
  static shmem_short_put_t shmem_short_put_handle = (shmem_short_put_t)NULL;
  if (!shmem_short_put_handle) {
    shmem_short_put_handle = get_function_handle("shmem_short_put");
  }

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
  static shmem_int_put_t shmem_int_put_handle = (shmem_int_put_t)NULL;
  if (!shmem_int_put_handle) {
    shmem_int_put_handle = get_function_handle("shmem_int_put");
  }

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
  static shmem_long_put_t shmem_long_put_handle = (shmem_long_put_t)NULL;
  if (!shmem_long_put_handle) {
    shmem_long_put_handle = get_function_handle("shmem_long_put");
  }

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
  static shmem_longlong_put_t shmem_longlong_put_handle = (shmem_longlong_put_t)NULL;
  if (!shmem_longlong_put_handle) {
    shmem_longlong_put_handle = get_function_handle("shmem_longlong_put");
  }

  shmem_longlong_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_put(long long * a1, const long long * a2, size_t a3, int a4) {
   __wrap_shmem_longlong_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float_put
 **********************************************************/

extern void  __wrap_shmem_float_put(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __real_shmem_float_put(float * a1, const float * a2, size_t a3, int a4)  {

  typedef void (*shmem_float_put_t)(float * a1, const float * a2, size_t a3, int a4);
  static shmem_float_put_t shmem_float_put_handle = (shmem_float_put_t)NULL;
  if (!shmem_float_put_handle) {
    shmem_float_put_handle = get_function_handle("shmem_float_put");
  }

  shmem_float_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float_put(float * a1, const float * a2, size_t a3, int a4) {
   __wrap_shmem_float_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_double_put
 **********************************************************/

extern void  __wrap_shmem_double_put(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __real_shmem_double_put(double * a1, const double * a2, size_t a3, int a4)  {

  typedef void (*shmem_double_put_t)(double * a1, const double * a2, size_t a3, int a4);
  static shmem_double_put_t shmem_double_put_handle = (shmem_double_put_t)NULL;
  if (!shmem_double_put_handle) {
    shmem_double_put_handle = get_function_handle("shmem_double_put");
  }

  shmem_double_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_double_put(double * a1, const double * a2, size_t a3, int a4) {
   __wrap_shmem_double_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float128_put
 **********************************************************/

extern void  __wrap_shmem_float128_put(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __real_shmem_float128_put(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  typedef void (*shmem_float128_put_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4);
  static shmem_float128_put_t shmem_float128_put_handle = (shmem_float128_put_t)NULL;
  if (!shmem_float128_put_handle) {
    shmem_float128_put_handle = get_function_handle("shmem_float128_put");
  }

  shmem_float128_put_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float128_put(__float128 * a1, const __float128 * a2, size_t a3, int a4) {
   __wrap_shmem_float128_put(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put16_signal
 **********************************************************/

extern void  __wrap_shmem_put16_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_put16_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_put16_signal_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_put16_signal_t shmem_put16_signal_handle = (shmem_put16_signal_t)NULL;
  if (!shmem_put16_signal_handle) {
    shmem_put16_signal_handle = get_function_handle("shmem_put16_signal");
  }

  shmem_put16_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_put16_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_put16_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_put32_signal
 **********************************************************/

extern void  __wrap_shmem_put32_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_put32_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_put32_signal_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_put32_signal_t shmem_put32_signal_handle = (shmem_put32_signal_t)NULL;
  if (!shmem_put32_signal_handle) {
    shmem_put32_signal_handle = get_function_handle("shmem_put32_signal");
  }

  shmem_put32_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_put32_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_put32_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_put64_signal
 **********************************************************/

extern void  __wrap_shmem_put64_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_put64_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_put64_signal_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_put64_signal_t shmem_put64_signal_handle = (shmem_put64_signal_t)NULL;
  if (!shmem_put64_signal_handle) {
    shmem_put64_signal_handle = get_function_handle("shmem_put64_signal");
  }

  shmem_put64_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_put64_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_put64_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_put128_signal
 **********************************************************/

extern void  __wrap_shmem_put128_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_put128_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_put128_signal_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_put128_signal_t shmem_put128_signal_handle = (shmem_put128_signal_t)NULL;
  if (!shmem_put128_signal_handle) {
    shmem_put128_signal_handle = get_function_handle("shmem_put128_signal");
  }

  shmem_put128_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_put128_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_put128_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_putmem_signal
 **********************************************************/

extern void  __wrap_shmem_putmem_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_putmem_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_putmem_signal_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_putmem_signal_t shmem_putmem_signal_handle = (shmem_putmem_signal_t)NULL;
  if (!shmem_putmem_signal_handle) {
    shmem_putmem_signal_handle = get_function_handle("shmem_putmem_signal");
  }

  shmem_putmem_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_putmem_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_putmem_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_short_put_signal
 **********************************************************/

extern void  __wrap_shmem_short_put_signal(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_short_put_signal(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_short_put_signal_t)(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_short_put_signal_t shmem_short_put_signal_handle = (shmem_short_put_signal_t)NULL;
  if (!shmem_short_put_signal_handle) {
    shmem_short_put_signal_handle = get_function_handle("shmem_short_put_signal");
  }

  shmem_short_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_short_put_signal(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_short_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_int_put_signal
 **********************************************************/

extern void  __wrap_shmem_int_put_signal(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_int_put_signal(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_int_put_signal_t)(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_int_put_signal_t shmem_int_put_signal_handle = (shmem_int_put_signal_t)NULL;
  if (!shmem_int_put_signal_handle) {
    shmem_int_put_signal_handle = get_function_handle("shmem_int_put_signal");
  }

  shmem_int_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_int_put_signal(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_int_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_long_put_signal
 **********************************************************/

extern void  __wrap_shmem_long_put_signal(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_long_put_signal(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_long_put_signal_t)(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_long_put_signal_t shmem_long_put_signal_handle = (shmem_long_put_signal_t)NULL;
  if (!shmem_long_put_signal_handle) {
    shmem_long_put_signal_handle = get_function_handle("shmem_long_put_signal");
  }

  shmem_long_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_long_put_signal(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_long_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_longlong_put_signal
 **********************************************************/

extern void  __wrap_shmem_longlong_put_signal(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_longlong_put_signal(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_longlong_put_signal_t)(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_longlong_put_signal_t shmem_longlong_put_signal_handle = (shmem_longlong_put_signal_t)NULL;
  if (!shmem_longlong_put_signal_handle) {
    shmem_longlong_put_signal_handle = get_function_handle("shmem_longlong_put_signal");
  }

  shmem_longlong_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longlong_put_signal(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_longlong_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float_put_signal
 **********************************************************/

extern void  __wrap_shmem_float_put_signal(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_float_put_signal(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_float_put_signal_t)(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_float_put_signal_t shmem_float_put_signal_handle = (shmem_float_put_signal_t)NULL;
  if (!shmem_float_put_signal_handle) {
    shmem_float_put_signal_handle = get_function_handle("shmem_float_put_signal");
  }

  shmem_float_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float_put_signal(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_float_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_double_put_signal
 **********************************************************/

extern void  __wrap_shmem_double_put_signal(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __real_shmem_double_put_signal(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  typedef void (*shmem_double_put_signal_t)(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6);
  static shmem_double_put_signal_t shmem_double_put_signal_handle = (shmem_double_put_signal_t)NULL;
  if (!shmem_double_put_signal_handle) {
    shmem_double_put_signal_handle = get_function_handle("shmem_double_put_signal");
  }

  shmem_double_put_signal_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_double_put_signal(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) {
   __wrap_shmem_double_put_signal(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_get16_nb
 **********************************************************/

extern void  __wrap_shmem_get16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_get16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_get16_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_get16_nb_t shmem_get16_nb_handle = (shmem_get16_nb_t)NULL;
  if (!shmem_get16_nb_handle) {
    shmem_get16_nb_handle = get_function_handle("shmem_get16_nb");
  }

  shmem_get16_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_get16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_get16_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_get32_nb
 **********************************************************/

extern void  __wrap_shmem_get32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_get32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_get32_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_get32_nb_t shmem_get32_nb_handle = (shmem_get32_nb_t)NULL;
  if (!shmem_get32_nb_handle) {
    shmem_get32_nb_handle = get_function_handle("shmem_get32_nb");
  }

  shmem_get32_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_get32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_get32_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_get64_nb
 **********************************************************/

extern void  __wrap_shmem_get64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_get64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_get64_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_get64_nb_t shmem_get64_nb_handle = (shmem_get64_nb_t)NULL;
  if (!shmem_get64_nb_handle) {
    shmem_get64_nb_handle = get_function_handle("shmem_get64_nb");
  }

  shmem_get64_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_get64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_get64_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_get128_nb
 **********************************************************/

extern void  __wrap_shmem_get128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_get128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_get128_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_get128_nb_t shmem_get128_nb_handle = (shmem_get128_nb_t)NULL;
  if (!shmem_get128_nb_handle) {
    shmem_get128_nb_handle = get_function_handle("shmem_get128_nb");
  }

  shmem_get128_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_get128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_get128_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_getmem_nb
 **********************************************************/

extern void  __wrap_shmem_getmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_getmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_getmem_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_getmem_nb_t shmem_getmem_nb_handle = (shmem_getmem_nb_t)NULL;
  if (!shmem_getmem_nb_handle) {
    shmem_getmem_nb_handle = get_function_handle("shmem_getmem_nb");
  }

  shmem_getmem_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_getmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_getmem_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_short_get_nb
 **********************************************************/

extern void  __wrap_shmem_short_get_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_short_get_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_short_get_nb_t)(short * a1, const short * a2, size_t a3, int a4, void ** a5);
  static shmem_short_get_nb_t shmem_short_get_nb_handle = (shmem_short_get_nb_t)NULL;
  if (!shmem_short_get_nb_handle) {
    shmem_short_get_nb_handle = get_function_handle("shmem_short_get_nb");
  }

  shmem_short_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_short_get_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_short_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_int_get_nb
 **********************************************************/

extern void  __wrap_shmem_int_get_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_int_get_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_int_get_nb_t)(int * a1, const int * a2, size_t a3, int a4, void ** a5);
  static shmem_int_get_nb_t shmem_int_get_nb_handle = (shmem_int_get_nb_t)NULL;
  if (!shmem_int_get_nb_handle) {
    shmem_int_get_nb_handle = get_function_handle("shmem_int_get_nb");
  }

  shmem_int_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_int_get_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_int_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_long_get_nb
 **********************************************************/

extern void  __wrap_shmem_long_get_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_long_get_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_long_get_nb_t)(long * a1, const long * a2, size_t a3, int a4, void ** a5);
  static shmem_long_get_nb_t shmem_long_get_nb_handle = (shmem_long_get_nb_t)NULL;
  if (!shmem_long_get_nb_handle) {
    shmem_long_get_nb_handle = get_function_handle("shmem_long_get_nb");
  }

  shmem_long_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_long_get_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_long_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_longlong_get_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_longlong_get_nb_t)(long long * a1, const long long * a2, size_t a3, int a4, void ** a5);
  static shmem_longlong_get_nb_t shmem_longlong_get_nb_handle = (shmem_longlong_get_nb_t)NULL;
  if (!shmem_longlong_get_nb_handle) {
    shmem_longlong_get_nb_handle = get_function_handle("shmem_longlong_get_nb");
  }

  shmem_longlong_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_longlong_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float_get_nb
 **********************************************************/

extern void  __wrap_shmem_float_get_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_float_get_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_float_get_nb_t)(float * a1, const float * a2, size_t a3, int a4, void ** a5);
  static shmem_float_get_nb_t shmem_float_get_nb_handle = (shmem_float_get_nb_t)NULL;
  if (!shmem_float_get_nb_handle) {
    shmem_float_get_nb_handle = get_function_handle("shmem_float_get_nb");
  }

  shmem_float_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float_get_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_float_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_double_get_nb
 **********************************************************/

extern void  __wrap_shmem_double_get_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_double_get_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_double_get_nb_t)(double * a1, const double * a2, size_t a3, int a4, void ** a5);
  static shmem_double_get_nb_t shmem_double_get_nb_handle = (shmem_double_get_nb_t)NULL;
  if (!shmem_double_get_nb_handle) {
    shmem_double_get_nb_handle = get_function_handle("shmem_double_get_nb");
  }

  shmem_double_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_double_get_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_double_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float128_get_nb
 **********************************************************/

extern void  __wrap_shmem_float128_get_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_float128_get_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_float128_get_nb_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5);
  static shmem_float128_get_nb_t shmem_float128_get_nb_handle = (shmem_float128_get_nb_t)NULL;
  if (!shmem_float128_get_nb_handle) {
    shmem_float128_get_nb_handle = get_function_handle("shmem_float128_get_nb");
  }

  shmem_float128_get_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float128_get_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_float128_get_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_put16_nb
 **********************************************************/

extern void  __wrap_shmem_put16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_put16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_put16_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_put16_nb_t shmem_put16_nb_handle = (shmem_put16_nb_t)NULL;
  if (!shmem_put16_nb_handle) {
    shmem_put16_nb_handle = get_function_handle("shmem_put16_nb");
  }

  shmem_put16_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_put16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_put16_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_put32_nb
 **********************************************************/

extern void  __wrap_shmem_put32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_put32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_put32_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_put32_nb_t shmem_put32_nb_handle = (shmem_put32_nb_t)NULL;
  if (!shmem_put32_nb_handle) {
    shmem_put32_nb_handle = get_function_handle("shmem_put32_nb");
  }

  shmem_put32_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_put32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_put32_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_put64_nb
 **********************************************************/

extern void  __wrap_shmem_put64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_put64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_put64_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_put64_nb_t shmem_put64_nb_handle = (shmem_put64_nb_t)NULL;
  if (!shmem_put64_nb_handle) {
    shmem_put64_nb_handle = get_function_handle("shmem_put64_nb");
  }

  shmem_put64_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_put64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_put64_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_put128_nb
 **********************************************************/

extern void  __wrap_shmem_put128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_put128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_put128_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_put128_nb_t shmem_put128_nb_handle = (shmem_put128_nb_t)NULL;
  if (!shmem_put128_nb_handle) {
    shmem_put128_nb_handle = get_function_handle("shmem_put128_nb");
  }

  shmem_put128_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_put128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_put128_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_putmem_nb
 **********************************************************/

extern void  __wrap_shmem_putmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_putmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_putmem_nb_t)(void * a1, const void * a2, size_t a3, int a4, void ** a5);
  static shmem_putmem_nb_t shmem_putmem_nb_handle = (shmem_putmem_nb_t)NULL;
  if (!shmem_putmem_nb_handle) {
    shmem_putmem_nb_handle = get_function_handle("shmem_putmem_nb");
  }

  shmem_putmem_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_putmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_putmem_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_short_put_nb
 **********************************************************/

extern void  __wrap_shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_short_put_nb_t)(short * a1, const short * a2, size_t a3, int a4, void ** a5);
  static shmem_short_put_nb_t shmem_short_put_nb_handle = (shmem_short_put_nb_t)NULL;
  if (!shmem_short_put_nb_handle) {
    shmem_short_put_nb_handle = get_function_handle("shmem_short_put_nb");
  }

  shmem_short_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_short_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_int_put_nb
 **********************************************************/

extern void  __wrap_shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_int_put_nb_t)(int * a1, const int * a2, size_t a3, int a4, void ** a5);
  static shmem_int_put_nb_t shmem_int_put_nb_handle = (shmem_int_put_nb_t)NULL;
  if (!shmem_int_put_nb_handle) {
    shmem_int_put_nb_handle = get_function_handle("shmem_int_put_nb");
  }

  shmem_int_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_int_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_long_put_nb
 **********************************************************/

extern void  __wrap_shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_long_put_nb_t)(long * a1, const long * a2, size_t a3, int a4, void ** a5);
  static shmem_long_put_nb_t shmem_long_put_nb_handle = (shmem_long_put_nb_t)NULL;
  if (!shmem_long_put_nb_handle) {
    shmem_long_put_nb_handle = get_function_handle("shmem_long_put_nb");
  }

  shmem_long_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_long_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_longlong_put_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_longlong_put_nb_t)(long long * a1, const long long * a2, size_t a3, int a4, void ** a5);
  static shmem_longlong_put_nb_t shmem_longlong_put_nb_handle = (shmem_longlong_put_nb_t)NULL;
  if (!shmem_longlong_put_nb_handle) {
    shmem_longlong_put_nb_handle = get_function_handle("shmem_longlong_put_nb");
  }

  shmem_longlong_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_longlong_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float_put_nb
 **********************************************************/

extern void  __wrap_shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_float_put_nb_t)(float * a1, const float * a2, size_t a3, int a4, void ** a5);
  static shmem_float_put_nb_t shmem_float_put_nb_handle = (shmem_float_put_nb_t)NULL;
  if (!shmem_float_put_nb_handle) {
    shmem_float_put_nb_handle = get_function_handle("shmem_float_put_nb");
  }

  shmem_float_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_float_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_double_put_nb
 **********************************************************/

extern void  __wrap_shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_double_put_nb_t)(double * a1, const double * a2, size_t a3, int a4, void ** a5);
  static shmem_double_put_nb_t shmem_double_put_nb_handle = (shmem_double_put_nb_t)NULL;
  if (!shmem_double_put_nb_handle) {
    shmem_double_put_nb_handle = get_function_handle("shmem_double_put_nb");
  }

  shmem_double_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_double_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float128_put_nb
 **********************************************************/

extern void  __wrap_shmem_float128_put_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) ;
extern void  __real_shmem_float128_put_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)  {

  typedef void (*shmem_float128_put_nb_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5);
  static shmem_float128_put_nb_t shmem_float128_put_nb_handle = (shmem_float128_put_nb_t)NULL;
  if (!shmem_float128_put_nb_handle) {
    shmem_float128_put_nb_handle = get_function_handle("shmem_float128_put_nb");
  }

  shmem_float128_put_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float128_put_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) {
   __wrap_shmem_float128_put_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_get8_nbi
 **********************************************************/

extern void  __wrap_shmem_get8_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get8_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get8_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_get8_nbi_t shmem_get8_nbi_handle = (shmem_get8_nbi_t)NULL;
  if (!shmem_get8_nbi_handle) {
    shmem_get8_nbi_handle = get_function_handle("shmem_get8_nbi");
  }

  shmem_get8_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get8_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get8_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get16_nbi
 **********************************************************/

extern void  __wrap_shmem_get16_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get16_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get16_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_get16_nbi_t shmem_get16_nbi_handle = (shmem_get16_nbi_t)NULL;
  if (!shmem_get16_nbi_handle) {
    shmem_get16_nbi_handle = get_function_handle("shmem_get16_nbi");
  }

  shmem_get16_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get16_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get16_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get32_nbi
 **********************************************************/

extern void  __wrap_shmem_get32_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get32_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get32_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_get32_nbi_t shmem_get32_nbi_handle = (shmem_get32_nbi_t)NULL;
  if (!shmem_get32_nbi_handle) {
    shmem_get32_nbi_handle = get_function_handle("shmem_get32_nbi");
  }

  shmem_get32_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get32_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get32_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get64_nbi
 **********************************************************/

extern void  __wrap_shmem_get64_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get64_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get64_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_get64_nbi_t shmem_get64_nbi_handle = (shmem_get64_nbi_t)NULL;
  if (!shmem_get64_nbi_handle) {
    shmem_get64_nbi_handle = get_function_handle("shmem_get64_nbi");
  }

  shmem_get64_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get64_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get64_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_get128_nbi
 **********************************************************/

extern void  __wrap_shmem_get128_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_get128_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_get128_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_get128_nbi_t shmem_get128_nbi_handle = (shmem_get128_nbi_t)NULL;
  if (!shmem_get128_nbi_handle) {
    shmem_get128_nbi_handle = get_function_handle("shmem_get128_nbi");
  }

  shmem_get128_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_get128_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_get128_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_getmem_nbi
 **********************************************************/

extern void  __wrap_shmem_getmem_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_getmem_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_getmem_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_getmem_nbi_t shmem_getmem_nbi_handle = (shmem_getmem_nbi_t)NULL;
  if (!shmem_getmem_nbi_handle) {
    shmem_getmem_nbi_handle = get_function_handle("shmem_getmem_nbi");
  }

  shmem_getmem_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_getmem_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_getmem_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_char_get_nbi
 **********************************************************/

extern void  __wrap_shmem_char_get_nbi(char * a1, const char * a2, size_t a3, int a4) ;
extern void  __real_shmem_char_get_nbi(char * a1, const char * a2, size_t a3, int a4)  {

  typedef void (*shmem_char_get_nbi_t)(char * a1, const char * a2, size_t a3, int a4);
  static shmem_char_get_nbi_t shmem_char_get_nbi_handle = (shmem_char_get_nbi_t)NULL;
  if (!shmem_char_get_nbi_handle) {
    shmem_char_get_nbi_handle = get_function_handle("shmem_char_get_nbi");
  }

  shmem_char_get_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_char_get_nbi(char * a1, const char * a2, size_t a3, int a4) {
   __wrap_shmem_char_get_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_get_nbi
 **********************************************************/

extern void  __wrap_shmem_short_get_nbi(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __real_shmem_short_get_nbi(short * a1, const short * a2, size_t a3, int a4)  {

  typedef void (*shmem_short_get_nbi_t)(short * a1, const short * a2, size_t a3, int a4);
  static shmem_short_get_nbi_t shmem_short_get_nbi_handle = (shmem_short_get_nbi_t)NULL;
  if (!shmem_short_get_nbi_handle) {
    shmem_short_get_nbi_handle = get_function_handle("shmem_short_get_nbi");
  }

  shmem_short_get_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_short_get_nbi(short * a1, const short * a2, size_t a3, int a4) {
   __wrap_shmem_short_get_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_get_nbi
 **********************************************************/

extern void  __wrap_shmem_int_get_nbi(int * a1, const int * a2, size_t a3, int a4) ;
extern void  __real_shmem_int_get_nbi(int * a1, const int * a2, size_t a3, int a4)  {

  typedef void (*shmem_int_get_nbi_t)(int * a1, const int * a2, size_t a3, int a4);
  static shmem_int_get_nbi_t shmem_int_get_nbi_handle = (shmem_int_get_nbi_t)NULL;
  if (!shmem_int_get_nbi_handle) {
    shmem_int_get_nbi_handle = get_function_handle("shmem_int_get_nbi");
  }

  shmem_int_get_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_int_get_nbi(int * a1, const int * a2, size_t a3, int a4) {
   __wrap_shmem_int_get_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_get_nbi
 **********************************************************/

extern void  __wrap_shmem_long_get_nbi(long * a1, const long * a2, size_t a3, int a4) ;
extern void  __real_shmem_long_get_nbi(long * a1, const long * a2, size_t a3, int a4)  {

  typedef void (*shmem_long_get_nbi_t)(long * a1, const long * a2, size_t a3, int a4);
  static shmem_long_get_nbi_t shmem_long_get_nbi_handle = (shmem_long_get_nbi_t)NULL;
  if (!shmem_long_get_nbi_handle) {
    shmem_long_get_nbi_handle = get_function_handle("shmem_long_get_nbi");
  }

  shmem_long_get_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_long_get_nbi(long * a1, const long * a2, size_t a3, int a4) {
   __wrap_shmem_long_get_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_get_nbi
 **********************************************************/

extern void  __wrap_shmem_longlong_get_nbi(long long * a1, const long long * a2, size_t a3, int a4) ;
extern void  __real_shmem_longlong_get_nbi(long long * a1, const long long * a2, size_t a3, int a4)  {

  typedef void (*shmem_longlong_get_nbi_t)(long long * a1, const long long * a2, size_t a3, int a4);
  static shmem_longlong_get_nbi_t shmem_longlong_get_nbi_handle = (shmem_longlong_get_nbi_t)NULL;
  if (!shmem_longlong_get_nbi_handle) {
    shmem_longlong_get_nbi_handle = get_function_handle("shmem_longlong_get_nbi");
  }

  shmem_longlong_get_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_get_nbi(long long * a1, const long long * a2, size_t a3, int a4) {
   __wrap_shmem_longlong_get_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float_get_nbi
 **********************************************************/

extern void  __wrap_shmem_float_get_nbi(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __real_shmem_float_get_nbi(float * a1, const float * a2, size_t a3, int a4)  {

  typedef void (*shmem_float_get_nbi_t)(float * a1, const float * a2, size_t a3, int a4);
  static shmem_float_get_nbi_t shmem_float_get_nbi_handle = (shmem_float_get_nbi_t)NULL;
  if (!shmem_float_get_nbi_handle) {
    shmem_float_get_nbi_handle = get_function_handle("shmem_float_get_nbi");
  }

  shmem_float_get_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float_get_nbi(float * a1, const float * a2, size_t a3, int a4) {
   __wrap_shmem_float_get_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_double_get_nbi
 **********************************************************/

extern void  __wrap_shmem_double_get_nbi(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __real_shmem_double_get_nbi(double * a1, const double * a2, size_t a3, int a4)  {

  typedef void (*shmem_double_get_nbi_t)(double * a1, const double * a2, size_t a3, int a4);
  static shmem_double_get_nbi_t shmem_double_get_nbi_handle = (shmem_double_get_nbi_t)NULL;
  if (!shmem_double_get_nbi_handle) {
    shmem_double_get_nbi_handle = get_function_handle("shmem_double_get_nbi");
  }

  shmem_double_get_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_double_get_nbi(double * a1, const double * a2, size_t a3, int a4) {
   __wrap_shmem_double_get_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float128_get_nbi
 **********************************************************/

extern void  __wrap_shmem_float128_get_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __real_shmem_float128_get_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  typedef void (*shmem_float128_get_nbi_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4);
  static shmem_float128_get_nbi_t shmem_float128_get_nbi_handle = (shmem_float128_get_nbi_t)NULL;
  if (!shmem_float128_get_nbi_handle) {
    shmem_float128_get_nbi_handle = get_function_handle("shmem_float128_get_nbi");
  }

  shmem_float128_get_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float128_get_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4) {
   __wrap_shmem_float128_get_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put8_nbi
 **********************************************************/

extern void  __wrap_shmem_put8_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put8_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put8_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_put8_nbi_t shmem_put8_nbi_handle = (shmem_put8_nbi_t)NULL;
  if (!shmem_put8_nbi_handle) {
    shmem_put8_nbi_handle = get_function_handle("shmem_put8_nbi");
  }

  shmem_put8_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put8_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put8_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put16_nbi
 **********************************************************/

extern void  __wrap_shmem_put16_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put16_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put16_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_put16_nbi_t shmem_put16_nbi_handle = (shmem_put16_nbi_t)NULL;
  if (!shmem_put16_nbi_handle) {
    shmem_put16_nbi_handle = get_function_handle("shmem_put16_nbi");
  }

  shmem_put16_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put16_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put16_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put32_nbi
 **********************************************************/

extern void  __wrap_shmem_put32_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put32_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put32_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_put32_nbi_t shmem_put32_nbi_handle = (shmem_put32_nbi_t)NULL;
  if (!shmem_put32_nbi_handle) {
    shmem_put32_nbi_handle = get_function_handle("shmem_put32_nbi");
  }

  shmem_put32_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put32_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put32_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put64_nbi
 **********************************************************/

extern void  __wrap_shmem_put64_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put64_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put64_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_put64_nbi_t shmem_put64_nbi_handle = (shmem_put64_nbi_t)NULL;
  if (!shmem_put64_nbi_handle) {
    shmem_put64_nbi_handle = get_function_handle("shmem_put64_nbi");
  }

  shmem_put64_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put64_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put64_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put128_nbi
 **********************************************************/

extern void  __wrap_shmem_put128_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_put128_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_put128_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_put128_nbi_t shmem_put128_nbi_handle = (shmem_put128_nbi_t)NULL;
  if (!shmem_put128_nbi_handle) {
    shmem_put128_nbi_handle = get_function_handle("shmem_put128_nbi");
  }

  shmem_put128_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_put128_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_put128_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_putmem_nbi
 **********************************************************/

extern void  __wrap_shmem_putmem_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __real_shmem_putmem_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  typedef void (*shmem_putmem_nbi_t)(void * a1, const void * a2, size_t a3, int a4);
  static shmem_putmem_nbi_t shmem_putmem_nbi_handle = (shmem_putmem_nbi_t)NULL;
  if (!shmem_putmem_nbi_handle) {
    shmem_putmem_nbi_handle = get_function_handle("shmem_putmem_nbi");
  }

  shmem_putmem_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_putmem_nbi(void * a1, const void * a2, size_t a3, int a4) {
   __wrap_shmem_putmem_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_char_put_nbi
 **********************************************************/

extern void  __wrap_shmem_char_put_nbi(char * a1, const char * a2, size_t a3, int a4) ;
extern void  __real_shmem_char_put_nbi(char * a1, const char * a2, size_t a3, int a4)  {

  typedef void (*shmem_char_put_nbi_t)(char * a1, const char * a2, size_t a3, int a4);
  static shmem_char_put_nbi_t shmem_char_put_nbi_handle = (shmem_char_put_nbi_t)NULL;
  if (!shmem_char_put_nbi_handle) {
    shmem_char_put_nbi_handle = get_function_handle("shmem_char_put_nbi");
  }

  shmem_char_put_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_char_put_nbi(char * a1, const char * a2, size_t a3, int a4) {
   __wrap_shmem_char_put_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_put_nbi
 **********************************************************/

extern void  __wrap_shmem_short_put_nbi(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __real_shmem_short_put_nbi(short * a1, const short * a2, size_t a3, int a4)  {

  typedef void (*shmem_short_put_nbi_t)(short * a1, const short * a2, size_t a3, int a4);
  static shmem_short_put_nbi_t shmem_short_put_nbi_handle = (shmem_short_put_nbi_t)NULL;
  if (!shmem_short_put_nbi_handle) {
    shmem_short_put_nbi_handle = get_function_handle("shmem_short_put_nbi");
  }

  shmem_short_put_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_short_put_nbi(short * a1, const short * a2, size_t a3, int a4) {
   __wrap_shmem_short_put_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_put_nbi
 **********************************************************/

extern void  __wrap_shmem_int_put_nbi(int * a1, const int * a2, size_t a3, int a4) ;
extern void  __real_shmem_int_put_nbi(int * a1, const int * a2, size_t a3, int a4)  {

  typedef void (*shmem_int_put_nbi_t)(int * a1, const int * a2, size_t a3, int a4);
  static shmem_int_put_nbi_t shmem_int_put_nbi_handle = (shmem_int_put_nbi_t)NULL;
  if (!shmem_int_put_nbi_handle) {
    shmem_int_put_nbi_handle = get_function_handle("shmem_int_put_nbi");
  }

  shmem_int_put_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_int_put_nbi(int * a1, const int * a2, size_t a3, int a4) {
   __wrap_shmem_int_put_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_put_nbi
 **********************************************************/

extern void  __wrap_shmem_long_put_nbi(long * a1, const long * a2, size_t a3, int a4) ;
extern void  __real_shmem_long_put_nbi(long * a1, const long * a2, size_t a3, int a4)  {

  typedef void (*shmem_long_put_nbi_t)(long * a1, const long * a2, size_t a3, int a4);
  static shmem_long_put_nbi_t shmem_long_put_nbi_handle = (shmem_long_put_nbi_t)NULL;
  if (!shmem_long_put_nbi_handle) {
    shmem_long_put_nbi_handle = get_function_handle("shmem_long_put_nbi");
  }

  shmem_long_put_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_long_put_nbi(long * a1, const long * a2, size_t a3, int a4) {
   __wrap_shmem_long_put_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_put_nbi
 **********************************************************/

extern void  __wrap_shmem_longlong_put_nbi(long long * a1, const long long * a2, size_t a3, int a4) ;
extern void  __real_shmem_longlong_put_nbi(long long * a1, const long long * a2, size_t a3, int a4)  {

  typedef void (*shmem_longlong_put_nbi_t)(long long * a1, const long long * a2, size_t a3, int a4);
  static shmem_longlong_put_nbi_t shmem_longlong_put_nbi_handle = (shmem_longlong_put_nbi_t)NULL;
  if (!shmem_longlong_put_nbi_handle) {
    shmem_longlong_put_nbi_handle = get_function_handle("shmem_longlong_put_nbi");
  }

  shmem_longlong_put_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_put_nbi(long long * a1, const long long * a2, size_t a3, int a4) {
   __wrap_shmem_longlong_put_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float_put_nbi
 **********************************************************/

extern void  __wrap_shmem_float_put_nbi(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __real_shmem_float_put_nbi(float * a1, const float * a2, size_t a3, int a4)  {

  typedef void (*shmem_float_put_nbi_t)(float * a1, const float * a2, size_t a3, int a4);
  static shmem_float_put_nbi_t shmem_float_put_nbi_handle = (shmem_float_put_nbi_t)NULL;
  if (!shmem_float_put_nbi_handle) {
    shmem_float_put_nbi_handle = get_function_handle("shmem_float_put_nbi");
  }

  shmem_float_put_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float_put_nbi(float * a1, const float * a2, size_t a3, int a4) {
   __wrap_shmem_float_put_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_double_put_nbi
 **********************************************************/

extern void  __wrap_shmem_double_put_nbi(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __real_shmem_double_put_nbi(double * a1, const double * a2, size_t a3, int a4)  {

  typedef void (*shmem_double_put_nbi_t)(double * a1, const double * a2, size_t a3, int a4);
  static shmem_double_put_nbi_t shmem_double_put_nbi_handle = (shmem_double_put_nbi_t)NULL;
  if (!shmem_double_put_nbi_handle) {
    shmem_double_put_nbi_handle = get_function_handle("shmem_double_put_nbi");
  }

  shmem_double_put_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_double_put_nbi(double * a1, const double * a2, size_t a3, int a4) {
   __wrap_shmem_double_put_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_float128_put_nbi
 **********************************************************/

extern void  __wrap_shmem_float128_put_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __real_shmem_float128_put_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  typedef void (*shmem_float128_put_nbi_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4);
  static shmem_float128_put_nbi_t shmem_float128_put_nbi_handle = (shmem_float128_put_nbi_t)NULL;
  if (!shmem_float128_put_nbi_handle) {
    shmem_float128_put_nbi_handle = get_function_handle("shmem_float128_put_nbi");
  }

  shmem_float128_put_nbi_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_float128_put_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4) {
   __wrap_shmem_float128_put_nbi(a1, a2, a3, a4);
}


/**********************************************************
   shmem_put16_signal_nb
 **********************************************************/

extern void  __wrap_shmem_put16_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_put16_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_put16_signal_nb_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_put16_signal_nb_t shmem_put16_signal_nb_handle = (shmem_put16_signal_nb_t)NULL;
  if (!shmem_put16_signal_nb_handle) {
    shmem_put16_signal_nb_handle = get_function_handle("shmem_put16_signal_nb");
  }

  shmem_put16_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_put16_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_put16_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_put32_signal_nb
 **********************************************************/

extern void  __wrap_shmem_put32_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_put32_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_put32_signal_nb_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_put32_signal_nb_t shmem_put32_signal_nb_handle = (shmem_put32_signal_nb_t)NULL;
  if (!shmem_put32_signal_nb_handle) {
    shmem_put32_signal_nb_handle = get_function_handle("shmem_put32_signal_nb");
  }

  shmem_put32_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_put32_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_put32_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_put64_signal_nb
 **********************************************************/

extern void  __wrap_shmem_put64_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_put64_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_put64_signal_nb_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_put64_signal_nb_t shmem_put64_signal_nb_handle = (shmem_put64_signal_nb_t)NULL;
  if (!shmem_put64_signal_nb_handle) {
    shmem_put64_signal_nb_handle = get_function_handle("shmem_put64_signal_nb");
  }

  shmem_put64_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_put64_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_put64_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_put128_signal_nb
 **********************************************************/

extern void  __wrap_shmem_put128_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_put128_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_put128_signal_nb_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_put128_signal_nb_t shmem_put128_signal_nb_handle = (shmem_put128_signal_nb_t)NULL;
  if (!shmem_put128_signal_nb_handle) {
    shmem_put128_signal_nb_handle = get_function_handle("shmem_put128_signal_nb");
  }

  shmem_put128_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_put128_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_put128_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_putmem_signal_nb
 **********************************************************/

extern void  __wrap_shmem_putmem_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_putmem_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_putmem_signal_nb_t)(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_putmem_signal_nb_t shmem_putmem_signal_nb_handle = (shmem_putmem_signal_nb_t)NULL;
  if (!shmem_putmem_signal_nb_handle) {
    shmem_putmem_signal_nb_handle = get_function_handle("shmem_putmem_signal_nb");
  }

  shmem_putmem_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_putmem_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_putmem_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_short_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_short_put_signal_nb(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_short_put_signal_nb(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_short_put_signal_nb_t)(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_short_put_signal_nb_t shmem_short_put_signal_nb_handle = (shmem_short_put_signal_nb_t)NULL;
  if (!shmem_short_put_signal_nb_handle) {
    shmem_short_put_signal_nb_handle = get_function_handle("shmem_short_put_signal_nb");
  }

  shmem_short_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_short_put_signal_nb(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_short_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_int_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_int_put_signal_nb(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_int_put_signal_nb(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_int_put_signal_nb_t)(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_int_put_signal_nb_t shmem_int_put_signal_nb_handle = (shmem_int_put_signal_nb_t)NULL;
  if (!shmem_int_put_signal_nb_handle) {
    shmem_int_put_signal_nb_handle = get_function_handle("shmem_int_put_signal_nb");
  }

  shmem_int_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_int_put_signal_nb(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_int_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_long_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_long_put_signal_nb(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_long_put_signal_nb(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_long_put_signal_nb_t)(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_long_put_signal_nb_t shmem_long_put_signal_nb_handle = (shmem_long_put_signal_nb_t)NULL;
  if (!shmem_long_put_signal_nb_handle) {
    shmem_long_put_signal_nb_handle = get_function_handle("shmem_long_put_signal_nb");
  }

  shmem_long_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_long_put_signal_nb(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_long_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_longlong_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_put_signal_nb(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_longlong_put_signal_nb(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_longlong_put_signal_nb_t)(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_longlong_put_signal_nb_t shmem_longlong_put_signal_nb_handle = (shmem_longlong_put_signal_nb_t)NULL;
  if (!shmem_longlong_put_signal_nb_handle) {
    shmem_longlong_put_signal_nb_handle = get_function_handle("shmem_longlong_put_signal_nb");
  }

  shmem_longlong_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_longlong_put_signal_nb(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_longlong_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_float_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_float_put_signal_nb(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_float_put_signal_nb(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_float_put_signal_nb_t)(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_float_put_signal_nb_t shmem_float_put_signal_nb_handle = (shmem_float_put_signal_nb_t)NULL;
  if (!shmem_float_put_signal_nb_handle) {
    shmem_float_put_signal_nb_handle = get_function_handle("shmem_float_put_signal_nb");
  }

  shmem_float_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_float_put_signal_nb(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_float_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_double_put_signal_nb
 **********************************************************/

extern void  __wrap_shmem_double_put_signal_nb(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __real_shmem_double_put_signal_nb(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  typedef void (*shmem_double_put_signal_nb_t)(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7);
  static shmem_double_put_signal_nb_t shmem_double_put_signal_nb_handle = (shmem_double_put_signal_nb_t)NULL;
  if (!shmem_double_put_signal_nb_handle) {
    shmem_double_put_signal_nb_handle = get_function_handle("shmem_double_put_signal_nb");
  }

  shmem_double_put_signal_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_double_put_signal_nb(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) {
   __wrap_shmem_double_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_char_iget
 **********************************************************/

extern void  __wrap_shmem_char_iget(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_char_iget(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_char_iget_t)(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_char_iget_t shmem_char_iget_handle = (shmem_char_iget_t)NULL;
  if (!shmem_char_iget_handle) {
    shmem_char_iget_handle = get_function_handle("shmem_char_iget");
  }

  shmem_char_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_char_iget(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_char_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_short_iget
 **********************************************************/

extern void  __wrap_shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_short_iget_t)(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_short_iget_t shmem_short_iget_handle = (shmem_short_iget_t)NULL;
  if (!shmem_short_iget_handle) {
    shmem_short_iget_handle = get_function_handle("shmem_short_iget");
  }

  shmem_short_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_short_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_int_iget
 **********************************************************/

extern void  __wrap_shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_int_iget_t)(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_int_iget_t shmem_int_iget_handle = (shmem_int_iget_t)NULL;
  if (!shmem_int_iget_handle) {
    shmem_int_iget_handle = get_function_handle("shmem_int_iget");
  }

  shmem_int_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_int_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_long_iget
 **********************************************************/

extern void  __wrap_shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_long_iget_t)(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_long_iget_t shmem_long_iget_handle = (shmem_long_iget_t)NULL;
  if (!shmem_long_iget_handle) {
    shmem_long_iget_handle = get_function_handle("shmem_long_iget");
  }

  shmem_long_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_long_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_longlong_iget
 **********************************************************/

extern void  __wrap_shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_longlong_iget_t)(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_longlong_iget_t shmem_longlong_iget_handle = (shmem_longlong_iget_t)NULL;
  if (!shmem_longlong_iget_handle) {
    shmem_longlong_iget_handle = get_function_handle("shmem_longlong_iget");
  }

  shmem_longlong_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_longlong_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float_iget
 **********************************************************/

extern void  __wrap_shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_float_iget_t)(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_float_iget_t shmem_float_iget_handle = (shmem_float_iget_t)NULL;
  if (!shmem_float_iget_handle) {
    shmem_float_iget_handle = get_function_handle("shmem_float_iget");
  }

  shmem_float_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_float_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_double_iget
 **********************************************************/

extern void  __wrap_shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_double_iget_t)(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_double_iget_t shmem_double_iget_handle = (shmem_double_iget_t)NULL;
  if (!shmem_double_iget_handle) {
    shmem_double_iget_handle = get_function_handle("shmem_double_iget");
  }

  shmem_double_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_double_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float128_iget
 **********************************************************/

extern void  __wrap_shmem_float128_iget(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_float128_iget(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_float128_iget_t)(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_float128_iget_t shmem_float128_iget_handle = (shmem_float128_iget_t)NULL;
  if (!shmem_float128_iget_handle) {
    shmem_float128_iget_handle = get_function_handle("shmem_float128_iget");
  }

  shmem_float128_iget_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float128_iget(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_float128_iget(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iget8
 **********************************************************/

extern void  __wrap_shmem_iget8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iget8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iget8_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_iget8_t shmem_iget8_handle = (shmem_iget8_t)NULL;
  if (!shmem_iget8_handle) {
    shmem_iget8_handle = get_function_handle("shmem_iget8");
  }

  shmem_iget8_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iget8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iget8(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iget16
 **********************************************************/

extern void  __wrap_shmem_iget16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iget16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iget16_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_iget16_t shmem_iget16_handle = (shmem_iget16_t)NULL;
  if (!shmem_iget16_handle) {
    shmem_iget16_handle = get_function_handle("shmem_iget16");
  }

  shmem_iget16_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iget16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iget16(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iget32
 **********************************************************/

extern void  __wrap_shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iget32_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_iget32_t shmem_iget32_handle = (shmem_iget32_t)NULL;
  if (!shmem_iget32_handle) {
    shmem_iget32_handle = get_function_handle("shmem_iget32");
  }

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
  static shmem_iget64_t shmem_iget64_handle = (shmem_iget64_t)NULL;
  if (!shmem_iget64_handle) {
    shmem_iget64_handle = get_function_handle("shmem_iget64");
  }

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
  static shmem_iget128_t shmem_iget128_handle = (shmem_iget128_t)NULL;
  if (!shmem_iget128_handle) {
    shmem_iget128_handle = get_function_handle("shmem_iget128");
  }

  shmem_iget128_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iget128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iget128(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_char_iput
 **********************************************************/

extern void  __wrap_shmem_char_iput(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_char_iput(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_char_iput_t)(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_char_iput_t shmem_char_iput_handle = (shmem_char_iput_t)NULL;
  if (!shmem_char_iput_handle) {
    shmem_char_iput_handle = get_function_handle("shmem_char_iput");
  }

  shmem_char_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_char_iput(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_char_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_short_iput
 **********************************************************/

extern void  __wrap_shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_short_iput_t)(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_short_iput_t shmem_short_iput_handle = (shmem_short_iput_t)NULL;
  if (!shmem_short_iput_handle) {
    shmem_short_iput_handle = get_function_handle("shmem_short_iput");
  }

  shmem_short_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_short_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_int_iput
 **********************************************************/

extern void  __wrap_shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_int_iput_t)(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_int_iput_t shmem_int_iput_handle = (shmem_int_iput_t)NULL;
  if (!shmem_int_iput_handle) {
    shmem_int_iput_handle = get_function_handle("shmem_int_iput");
  }

  shmem_int_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_int_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_long_iput
 **********************************************************/

extern void  __wrap_shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_long_iput_t)(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_long_iput_t shmem_long_iput_handle = (shmem_long_iput_t)NULL;
  if (!shmem_long_iput_handle) {
    shmem_long_iput_handle = get_function_handle("shmem_long_iput");
  }

  shmem_long_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_long_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_longlong_iput
 **********************************************************/

extern void  __wrap_shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_longlong_iput_t)(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_longlong_iput_t shmem_longlong_iput_handle = (shmem_longlong_iput_t)NULL;
  if (!shmem_longlong_iput_handle) {
    shmem_longlong_iput_handle = get_function_handle("shmem_longlong_iput");
  }

  shmem_longlong_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_longlong_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float_iput
 **********************************************************/

extern void  __wrap_shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_float_iput_t)(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_float_iput_t shmem_float_iput_handle = (shmem_float_iput_t)NULL;
  if (!shmem_float_iput_handle) {
    shmem_float_iput_handle = get_function_handle("shmem_float_iput");
  }

  shmem_float_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_float_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_double_iput
 **********************************************************/

extern void  __wrap_shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_double_iput_t)(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_double_iput_t shmem_double_iput_handle = (shmem_double_iput_t)NULL;
  if (!shmem_double_iput_handle) {
    shmem_double_iput_handle = get_function_handle("shmem_double_iput");
  }

  shmem_double_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_double_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_float128_iput
 **********************************************************/

extern void  __wrap_shmem_float128_iput(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_float128_iput(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_float128_iput_t)(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_float128_iput_t shmem_float128_iput_handle = (shmem_float128_iput_t)NULL;
  if (!shmem_float128_iput_handle) {
    shmem_float128_iput_handle = get_function_handle("shmem_float128_iput");
  }

  shmem_float128_iput_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_float128_iput(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_float128_iput(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iput8
 **********************************************************/

extern void  __wrap_shmem_iput8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iput8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iput8_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_iput8_t shmem_iput8_handle = (shmem_iput8_t)NULL;
  if (!shmem_iput8_handle) {
    shmem_iput8_handle = get_function_handle("shmem_iput8");
  }

  shmem_iput8_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iput8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iput8(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iput16
 **********************************************************/

extern void  __wrap_shmem_iput16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iput16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iput16_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_iput16_t shmem_iput16_handle = (shmem_iput16_t)NULL;
  if (!shmem_iput16_handle) {
    shmem_iput16_handle = get_function_handle("shmem_iput16");
  }

  shmem_iput16_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iput16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iput16(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_iput32
 **********************************************************/

extern void  __wrap_shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __real_shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  typedef void (*shmem_iput32_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6);
  static shmem_iput32_t shmem_iput32_handle = (shmem_iput32_t)NULL;
  if (!shmem_iput32_handle) {
    shmem_iput32_handle = get_function_handle("shmem_iput32");
  }

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
  static shmem_iput64_t shmem_iput64_handle = (shmem_iput64_t)NULL;
  if (!shmem_iput64_handle) {
    shmem_iput64_handle = get_function_handle("shmem_iput64");
  }

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
  static shmem_iput128_t shmem_iput128_handle = (shmem_iput128_t)NULL;
  if (!shmem_iput128_handle) {
    shmem_iput128_handle = get_function_handle("shmem_iput128");
  }

  shmem_iput128_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_iput128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {
   __wrap_shmem_iput128(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_char_g
 **********************************************************/

extern char  __wrap_shmem_char_g(const char * a1, int a2) ;
extern char  __real_shmem_char_g(const char * a1, int a2)  {

  char retval;
  typedef char (*shmem_char_g_t)(const char * a1, int a2);
  static shmem_char_g_t shmem_char_g_handle = (shmem_char_g_t)NULL;
  if (!shmem_char_g_handle) {
    shmem_char_g_handle = get_function_handle("shmem_char_g");
  }

  retval  =  shmem_char_g_handle ( a1,  a2);
  return retval;

}

extern char  shmem_char_g(const char * a1, int a2) {
   __wrap_shmem_char_g(a1, a2);
}


/**********************************************************
   shmem_short_g
 **********************************************************/

extern short  __wrap_shmem_short_g(const short * a1, int a2) ;
extern short  __real_shmem_short_g(const short * a1, int a2)  {

  short retval;
  typedef short (*shmem_short_g_t)(const short * a1, int a2);
  static shmem_short_g_t shmem_short_g_handle = (shmem_short_g_t)NULL;
  if (!shmem_short_g_handle) {
    shmem_short_g_handle = get_function_handle("shmem_short_g");
  }

  retval  =  shmem_short_g_handle ( a1,  a2);
  return retval;

}

extern short  shmem_short_g(const short * a1, int a2) {
   __wrap_shmem_short_g(a1, a2);
}


/**********************************************************
   shmem_int_g
 **********************************************************/

extern int  __wrap_shmem_int_g(const int * a1, int a2) ;
extern int  __real_shmem_int_g(const int * a1, int a2)  {

  int retval;
  typedef int (*shmem_int_g_t)(const int * a1, int a2);
  static shmem_int_g_t shmem_int_g_handle = (shmem_int_g_t)NULL;
  if (!shmem_int_g_handle) {
    shmem_int_g_handle = get_function_handle("shmem_int_g");
  }

  retval  =  shmem_int_g_handle ( a1,  a2);
  return retval;

}

extern int  shmem_int_g(const int * a1, int a2) {
   __wrap_shmem_int_g(a1, a2);
}


/**********************************************************
   shmem_long_g
 **********************************************************/

extern long  __wrap_shmem_long_g(const long * a1, int a2) ;
extern long  __real_shmem_long_g(const long * a1, int a2)  {

  long retval;
  typedef long (*shmem_long_g_t)(const long * a1, int a2);
  static shmem_long_g_t shmem_long_g_handle = (shmem_long_g_t)NULL;
  if (!shmem_long_g_handle) {
    shmem_long_g_handle = get_function_handle("shmem_long_g");
  }

  retval  =  shmem_long_g_handle ( a1,  a2);
  return retval;

}

extern long  shmem_long_g(const long * a1, int a2) {
   __wrap_shmem_long_g(a1, a2);
}


/**********************************************************
   shmem_longlong_g
 **********************************************************/

extern long long  __wrap_shmem_longlong_g(const long long * a1, int a2) ;
extern long long  __real_shmem_longlong_g(const long long * a1, int a2)  {

  long long retval;
  typedef long long (*shmem_longlong_g_t)(const long long * a1, int a2);
  static shmem_longlong_g_t shmem_longlong_g_handle = (shmem_longlong_g_t)NULL;
  if (!shmem_longlong_g_handle) {
    shmem_longlong_g_handle = get_function_handle("shmem_longlong_g");
  }

  retval  =  shmem_longlong_g_handle ( a1,  a2);
  return retval;

}

extern long long  shmem_longlong_g(const long long * a1, int a2) {
   __wrap_shmem_longlong_g(a1, a2);
}


/**********************************************************
   shmem_float_g
 **********************************************************/

extern float  __wrap_shmem_float_g(const float * a1, int a2) ;
extern float  __real_shmem_float_g(const float * a1, int a2)  {

  float retval;
  typedef float (*shmem_float_g_t)(const float * a1, int a2);
  static shmem_float_g_t shmem_float_g_handle = (shmem_float_g_t)NULL;
  if (!shmem_float_g_handle) {
    shmem_float_g_handle = get_function_handle("shmem_float_g");
  }

  retval  =  shmem_float_g_handle ( a1,  a2);
  return retval;

}

extern float  shmem_float_g(const float * a1, int a2) {
   __wrap_shmem_float_g(a1, a2);
}


/**********************************************************
   shmem_double_g
 **********************************************************/

extern double  __wrap_shmem_double_g(const double * a1, int a2) ;
extern double  __real_shmem_double_g(const double * a1, int a2)  {

  double retval;
  typedef double (*shmem_double_g_t)(const double * a1, int a2);
  static shmem_double_g_t shmem_double_g_handle = (shmem_double_g_t)NULL;
  if (!shmem_double_g_handle) {
    shmem_double_g_handle = get_function_handle("shmem_double_g");
  }

  retval  =  shmem_double_g_handle ( a1,  a2);
  return retval;

}

extern double  shmem_double_g(const double * a1, int a2) {
   __wrap_shmem_double_g(a1, a2);
}


/**********************************************************
   shmem_ld80_g
 **********************************************************/

extern long double  __wrap_shmem_ld80_g(const long double * a1, int a2) ;
extern long double  __real_shmem_ld80_g(const long double * a1, int a2)  {

  long double retval;
  typedef long double (*shmem_ld80_g_t)(const long double * a1, int a2);
  static shmem_ld80_g_t shmem_ld80_g_handle = (shmem_ld80_g_t)NULL;
  if (!shmem_ld80_g_handle) {
    shmem_ld80_g_handle = get_function_handle("shmem_ld80_g");
  }

  retval  =  shmem_ld80_g_handle ( a1,  a2);
  return retval;

}

extern long double  shmem_ld80_g(const long double * a1, int a2) {
   __wrap_shmem_ld80_g(a1, a2);
}


/**********************************************************
   shmem_float128_g
 **********************************************************/

extern __float128  __wrap_shmem_float128_g(const __float128 * a1, int a2) ;
extern __float128  __real_shmem_float128_g(const __float128 * a1, int a2)  {

  __float128 retval;
  typedef __float128 (*shmem_float128_g_t)(const __float128 * a1, int a2);
  static shmem_float128_g_t shmem_float128_g_handle = (shmem_float128_g_t)NULL;
  if (!shmem_float128_g_handle) {
    shmem_float128_g_handle = get_function_handle("shmem_float128_g");
  }

  retval  =  shmem_float128_g_handle ( a1,  a2);
  return retval;

}

extern __float128  shmem_float128_g(const __float128 * a1, int a2) {
   __wrap_shmem_float128_g(a1, a2);
}


/**********************************************************
   shmem_char_p
 **********************************************************/

extern void  __wrap_shmem_char_p(char * a1, char a2, int a3) ;
extern void  __real_shmem_char_p(char * a1, char a2, int a3)  {

  typedef void (*shmem_char_p_t)(char * a1, char a2, int a3);
  static shmem_char_p_t shmem_char_p_handle = (shmem_char_p_t)NULL;
  if (!shmem_char_p_handle) {
    shmem_char_p_handle = get_function_handle("shmem_char_p");
  }

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
  static shmem_short_p_t shmem_short_p_handle = (shmem_short_p_t)NULL;
  if (!shmem_short_p_handle) {
    shmem_short_p_handle = get_function_handle("shmem_short_p");
  }

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
  static shmem_int_p_t shmem_int_p_handle = (shmem_int_p_t)NULL;
  if (!shmem_int_p_handle) {
    shmem_int_p_handle = get_function_handle("shmem_int_p");
  }

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
  static shmem_long_p_t shmem_long_p_handle = (shmem_long_p_t)NULL;
  if (!shmem_long_p_handle) {
    shmem_long_p_handle = get_function_handle("shmem_long_p");
  }

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
  static shmem_longlong_p_t shmem_longlong_p_handle = (shmem_longlong_p_t)NULL;
  if (!shmem_longlong_p_handle) {
    shmem_longlong_p_handle = get_function_handle("shmem_longlong_p");
  }

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
  static shmem_float_p_t shmem_float_p_handle = (shmem_float_p_t)NULL;
  if (!shmem_float_p_handle) {
    shmem_float_p_handle = get_function_handle("shmem_float_p");
  }

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
  static shmem_double_p_t shmem_double_p_handle = (shmem_double_p_t)NULL;
  if (!shmem_double_p_handle) {
    shmem_double_p_handle = get_function_handle("shmem_double_p");
  }

  shmem_double_p_handle ( a1,  a2,  a3);

}

extern void  shmem_double_p(double * a1, double a2, int a3) {
   __wrap_shmem_double_p(a1, a2, a3);
}


/**********************************************************
   shmem_ld80_p
 **********************************************************/

extern void  __wrap_shmem_ld80_p(long double * a1, long double a2, int a3) ;
extern void  __real_shmem_ld80_p(long double * a1, long double a2, int a3)  {

  typedef void (*shmem_ld80_p_t)(long double * a1, long double a2, int a3);
  static shmem_ld80_p_t shmem_ld80_p_handle = (shmem_ld80_p_t)NULL;
  if (!shmem_ld80_p_handle) {
    shmem_ld80_p_handle = get_function_handle("shmem_ld80_p");
  }

  shmem_ld80_p_handle ( a1,  a2,  a3);

}

extern void  shmem_ld80_p(long double * a1, long double a2, int a3) {
   __wrap_shmem_ld80_p(a1, a2, a3);
}


/**********************************************************
   shmem_float128_p
 **********************************************************/

extern void  __wrap_shmem_float128_p(__float128 * a1, __float128 a2, int a3) ;
extern void  __real_shmem_float128_p(__float128 * a1, __float128 a2, int a3)  {

  typedef void (*shmem_float128_p_t)(__float128 * a1, __float128 a2, int a3);
  static shmem_float128_p_t shmem_float128_p_handle = (shmem_float128_p_t)NULL;
  if (!shmem_float128_p_handle) {
    shmem_float128_p_handle = get_function_handle("shmem_float128_p");
  }

  shmem_float128_p_handle ( a1,  a2,  a3);

}

extern void  shmem_float128_p(__float128 * a1, __float128 a2, int a3) {
   __wrap_shmem_float128_p(a1, a2, a3);
}


/**********************************************************
   shmem_short_swap
 **********************************************************/

extern short  __wrap_shmem_short_swap(short * a1, short a2, int a3) ;
extern short  __real_shmem_short_swap(short * a1, short a2, int a3)  {

  short retval;
  typedef short (*shmem_short_swap_t)(short * a1, short a2, int a3);
  static shmem_short_swap_t shmem_short_swap_handle = (shmem_short_swap_t)NULL;
  if (!shmem_short_swap_handle) {
    shmem_short_swap_handle = get_function_handle("shmem_short_swap");
  }

  retval  =  shmem_short_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern short  shmem_short_swap(short * a1, short a2, int a3) {
   __wrap_shmem_short_swap(a1, a2, a3);
}


/**********************************************************
   shmem_int_swap
 **********************************************************/

extern int  __wrap_shmem_int_swap(int * a1, int a2, int a3) ;
extern int  __real_shmem_int_swap(int * a1, int a2, int a3)  {

  int retval;
  typedef int (*shmem_int_swap_t)(int * a1, int a2, int a3);
  static shmem_int_swap_t shmem_int_swap_handle = (shmem_int_swap_t)NULL;
  if (!shmem_int_swap_handle) {
    shmem_int_swap_handle = get_function_handle("shmem_int_swap");
  }

  retval  =  shmem_int_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern int  shmem_int_swap(int * a1, int a2, int a3) {
   __wrap_shmem_int_swap(a1, a2, a3);
}


/**********************************************************
   shmem_long_swap
 **********************************************************/

extern long  __wrap_shmem_long_swap(long * a1, long a2, int a3) ;
extern long  __real_shmem_long_swap(long * a1, long a2, int a3)  {

  long retval;
  typedef long (*shmem_long_swap_t)(long * a1, long a2, int a3);
  static shmem_long_swap_t shmem_long_swap_handle = (shmem_long_swap_t)NULL;
  if (!shmem_long_swap_handle) {
    shmem_long_swap_handle = get_function_handle("shmem_long_swap");
  }

  retval  =  shmem_long_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern long  shmem_long_swap(long * a1, long a2, int a3) {
   __wrap_shmem_long_swap(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_swap
 **********************************************************/

extern long long  __wrap_shmem_longlong_swap(long long * a1, long long a2, int a3) ;
extern long long  __real_shmem_longlong_swap(long long * a1, long long a2, int a3)  {

  long long retval;
  typedef long long (*shmem_longlong_swap_t)(long long * a1, long long a2, int a3);
  static shmem_longlong_swap_t shmem_longlong_swap_handle = (shmem_longlong_swap_t)NULL;
  if (!shmem_longlong_swap_handle) {
    shmem_longlong_swap_handle = get_function_handle("shmem_longlong_swap");
  }

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
  static shmem_float_swap_t shmem_float_swap_handle = (shmem_float_swap_t)NULL;
  if (!shmem_float_swap_handle) {
    shmem_float_swap_handle = get_function_handle("shmem_float_swap");
  }

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
  static shmem_double_swap_t shmem_double_swap_handle = (shmem_double_swap_t)NULL;
  if (!shmem_double_swap_handle) {
    shmem_double_swap_handle = get_function_handle("shmem_double_swap");
  }

  retval  =  shmem_double_swap_handle ( a1,  a2,  a3);
  return retval;

}

extern double  shmem_double_swap(double * a1, double a2, int a3) {
   __wrap_shmem_double_swap(a1, a2, a3);
}


/**********************************************************
   shmem_short_swap_nb
 **********************************************************/

extern void  __wrap_shmem_short_swap_nb(short * a1, short * a2, short a3, int a4, void ** a5) ;
extern void  __real_shmem_short_swap_nb(short * a1, short * a2, short a3, int a4, void ** a5)  {

  typedef void (*shmem_short_swap_nb_t)(short * a1, short * a2, short a3, int a4, void ** a5);
  static shmem_short_swap_nb_t shmem_short_swap_nb_handle = (shmem_short_swap_nb_t)NULL;
  if (!shmem_short_swap_nb_handle) {
    shmem_short_swap_nb_handle = get_function_handle("shmem_short_swap_nb");
  }

  shmem_short_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_short_swap_nb(short * a1, short * a2, short a3, int a4, void ** a5) {
   __wrap_shmem_short_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_int_swap_nb
 **********************************************************/

extern void  __wrap_shmem_int_swap_nb(int * a1, int * a2, int a3, int a4, void ** a5) ;
extern void  __real_shmem_int_swap_nb(int * a1, int * a2, int a3, int a4, void ** a5)  {

  typedef void (*shmem_int_swap_nb_t)(int * a1, int * a2, int a3, int a4, void ** a5);
  static shmem_int_swap_nb_t shmem_int_swap_nb_handle = (shmem_int_swap_nb_t)NULL;
  if (!shmem_int_swap_nb_handle) {
    shmem_int_swap_nb_handle = get_function_handle("shmem_int_swap_nb");
  }

  shmem_int_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_int_swap_nb(int * a1, int * a2, int a3, int a4, void ** a5) {
   __wrap_shmem_int_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_long_swap_nb
 **********************************************************/

extern void  __wrap_shmem_long_swap_nb(long * a1, long * a2, long a3, int a4, void ** a5) ;
extern void  __real_shmem_long_swap_nb(long * a1, long * a2, long a3, int a4, void ** a5)  {

  typedef void (*shmem_long_swap_nb_t)(long * a1, long * a2, long a3, int a4, void ** a5);
  static shmem_long_swap_nb_t shmem_long_swap_nb_handle = (shmem_long_swap_nb_t)NULL;
  if (!shmem_long_swap_nb_handle) {
    shmem_long_swap_nb_handle = get_function_handle("shmem_long_swap_nb");
  }

  shmem_long_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_long_swap_nb(long * a1, long * a2, long a3, int a4, void ** a5) {
   __wrap_shmem_long_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_longlong_swap_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_swap_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) ;
extern void  __real_shmem_longlong_swap_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5)  {

  typedef void (*shmem_longlong_swap_nb_t)(long long * a1, long long * a2, long long a3, int a4, void ** a5);
  static shmem_longlong_swap_nb_t shmem_longlong_swap_nb_handle = (shmem_longlong_swap_nb_t)NULL;
  if (!shmem_longlong_swap_nb_handle) {
    shmem_longlong_swap_nb_handle = get_function_handle("shmem_longlong_swap_nb");
  }

  shmem_longlong_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_longlong_swap_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) {
   __wrap_shmem_longlong_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_float_swap_nb
 **********************************************************/

extern void  __wrap_shmem_float_swap_nb(float * a1, float * a2, float a3, int a4, void ** a5) ;
extern void  __real_shmem_float_swap_nb(float * a1, float * a2, float a3, int a4, void ** a5)  {

  typedef void (*shmem_float_swap_nb_t)(float * a1, float * a2, float a3, int a4, void ** a5);
  static shmem_float_swap_nb_t shmem_float_swap_nb_handle = (shmem_float_swap_nb_t)NULL;
  if (!shmem_float_swap_nb_handle) {
    shmem_float_swap_nb_handle = get_function_handle("shmem_float_swap_nb");
  }

  shmem_float_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_float_swap_nb(float * a1, float * a2, float a3, int a4, void ** a5) {
   __wrap_shmem_float_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_double_swap_nb
 **********************************************************/

extern void  __wrap_shmem_double_swap_nb(double * a1, double * a2, double a3, int a4, void ** a5) ;
extern void  __real_shmem_double_swap_nb(double * a1, double * a2, double a3, int a4, void ** a5)  {

  typedef void (*shmem_double_swap_nb_t)(double * a1, double * a2, double a3, int a4, void ** a5);
  static shmem_double_swap_nb_t shmem_double_swap_nb_handle = (shmem_double_swap_nb_t)NULL;
  if (!shmem_double_swap_nb_handle) {
    shmem_double_swap_nb_handle = get_function_handle("shmem_double_swap_nb");
  }

  shmem_double_swap_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_double_swap_nb(double * a1, double * a2, double a3, int a4, void ** a5) {
   __wrap_shmem_double_swap_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_short_cswap
 **********************************************************/

extern short  __wrap_shmem_short_cswap(short * a1, short a2, short a3, int a4) ;
extern short  __real_shmem_short_cswap(short * a1, short a2, short a3, int a4)  {

  short retval;
  typedef short (*shmem_short_cswap_t)(short * a1, short a2, short a3, int a4);
  static shmem_short_cswap_t shmem_short_cswap_handle = (shmem_short_cswap_t)NULL;
  if (!shmem_short_cswap_handle) {
    shmem_short_cswap_handle = get_function_handle("shmem_short_cswap");
  }

  retval  =  shmem_short_cswap_handle ( a1,  a2,  a3,  a4);
  return retval;

}

extern short  shmem_short_cswap(short * a1, short a2, short a3, int a4) {
   __wrap_shmem_short_cswap(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_cswap
 **********************************************************/

extern int  __wrap_shmem_int_cswap(int * a1, int a2, int a3, int a4) ;
extern int  __real_shmem_int_cswap(int * a1, int a2, int a3, int a4)  {

  int retval;
  typedef int (*shmem_int_cswap_t)(int * a1, int a2, int a3, int a4);
  static shmem_int_cswap_t shmem_int_cswap_handle = (shmem_int_cswap_t)NULL;
  if (!shmem_int_cswap_handle) {
    shmem_int_cswap_handle = get_function_handle("shmem_int_cswap");
  }

  retval  =  shmem_int_cswap_handle ( a1,  a2,  a3,  a4);
  return retval;

}

extern int  shmem_int_cswap(int * a1, int a2, int a3, int a4) {
   __wrap_shmem_int_cswap(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_cswap
 **********************************************************/

extern long  __wrap_shmem_long_cswap(long * a1, long a2, long a3, int a4) ;
extern long  __real_shmem_long_cswap(long * a1, long a2, long a3, int a4)  {

  long retval;
  typedef long (*shmem_long_cswap_t)(long * a1, long a2, long a3, int a4);
  static shmem_long_cswap_t shmem_long_cswap_handle = (shmem_long_cswap_t)NULL;
  if (!shmem_long_cswap_handle) {
    shmem_long_cswap_handle = get_function_handle("shmem_long_cswap");
  }

  retval  =  shmem_long_cswap_handle ( a1,  a2,  a3,  a4);
  return retval;

}

extern long  shmem_long_cswap(long * a1, long a2, long a3, int a4) {
   __wrap_shmem_long_cswap(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_cswap
 **********************************************************/

extern long long  __wrap_shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4) ;
extern long long  __real_shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4)  {

  long long retval;
  typedef long long (*shmem_longlong_cswap_t)(long long * a1, long long a2, long long a3, int a4);
  static shmem_longlong_cswap_t shmem_longlong_cswap_handle = (shmem_longlong_cswap_t)NULL;
  if (!shmem_longlong_cswap_handle) {
    shmem_longlong_cswap_handle = get_function_handle("shmem_longlong_cswap");
  }

  retval  =  shmem_longlong_cswap_handle ( a1,  a2,  a3,  a4);
  return retval;

}

extern long long  shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4) {
   __wrap_shmem_longlong_cswap(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_cswap_nb
 **********************************************************/

extern void  __wrap_shmem_short_cswap_nb(short * a1, short * a2, short a3, short a4, int a5, void ** a6) ;
extern void  __real_shmem_short_cswap_nb(short * a1, short * a2, short a3, short a4, int a5, void ** a6)  {

  typedef void (*shmem_short_cswap_nb_t)(short * a1, short * a2, short a3, short a4, int a5, void ** a6);
  static shmem_short_cswap_nb_t shmem_short_cswap_nb_handle = (shmem_short_cswap_nb_t)NULL;
  if (!shmem_short_cswap_nb_handle) {
    shmem_short_cswap_nb_handle = get_function_handle("shmem_short_cswap_nb");
  }

  shmem_short_cswap_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_short_cswap_nb(short * a1, short * a2, short a3, short a4, int a5, void ** a6) {
   __wrap_shmem_short_cswap_nb(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_int_cswap_nb
 **********************************************************/

extern void  __wrap_shmem_int_cswap_nb(int * a1, int * a2, int a3, int a4, int a5, void ** a6) ;
extern void  __real_shmem_int_cswap_nb(int * a1, int * a2, int a3, int a4, int a5, void ** a6)  {

  typedef void (*shmem_int_cswap_nb_t)(int * a1, int * a2, int a3, int a4, int a5, void ** a6);
  static shmem_int_cswap_nb_t shmem_int_cswap_nb_handle = (shmem_int_cswap_nb_t)NULL;
  if (!shmem_int_cswap_nb_handle) {
    shmem_int_cswap_nb_handle = get_function_handle("shmem_int_cswap_nb");
  }

  shmem_int_cswap_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_int_cswap_nb(int * a1, int * a2, int a3, int a4, int a5, void ** a6) {
   __wrap_shmem_int_cswap_nb(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_long_cswap_nb
 **********************************************************/

extern void  __wrap_shmem_long_cswap_nb(long * a1, long * a2, long a3, long a4, int a5, void ** a6) ;
extern void  __real_shmem_long_cswap_nb(long * a1, long * a2, long a3, long a4, int a5, void ** a6)  {

  typedef void (*shmem_long_cswap_nb_t)(long * a1, long * a2, long a3, long a4, int a5, void ** a6);
  static shmem_long_cswap_nb_t shmem_long_cswap_nb_handle = (shmem_long_cswap_nb_t)NULL;
  if (!shmem_long_cswap_nb_handle) {
    shmem_long_cswap_nb_handle = get_function_handle("shmem_long_cswap_nb");
  }

  shmem_long_cswap_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_long_cswap_nb(long * a1, long * a2, long a3, long a4, int a5, void ** a6) {
   __wrap_shmem_long_cswap_nb(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_longlong_cswap_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_cswap_nb(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6) ;
extern void  __real_shmem_longlong_cswap_nb(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6)  {

  typedef void (*shmem_longlong_cswap_nb_t)(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6);
  static shmem_longlong_cswap_nb_t shmem_longlong_cswap_nb_handle = (shmem_longlong_cswap_nb_t)NULL;
  if (!shmem_longlong_cswap_nb_handle) {
    shmem_longlong_cswap_nb_handle = get_function_handle("shmem_longlong_cswap_nb");
  }

  shmem_longlong_cswap_nb_handle ( a1,  a2,  a3,  a4,  a5,  a6);

}

extern void  shmem_longlong_cswap_nb(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6) {
   __wrap_shmem_longlong_cswap_nb(a1, a2, a3, a4, a5, a6);
}


/**********************************************************
   shmem_short_finc
 **********************************************************/

extern short  __wrap_shmem_short_finc(short * a1, int a2) ;
extern short  __real_shmem_short_finc(short * a1, int a2)  {

  short retval;
  typedef short (*shmem_short_finc_t)(short * a1, int a2);
  static shmem_short_finc_t shmem_short_finc_handle = (shmem_short_finc_t)NULL;
  if (!shmem_short_finc_handle) {
    shmem_short_finc_handle = get_function_handle("shmem_short_finc");
  }

  retval  =  shmem_short_finc_handle ( a1,  a2);
  return retval;

}

extern short  shmem_short_finc(short * a1, int a2) {
   __wrap_shmem_short_finc(a1, a2);
}


/**********************************************************
   shmem_int_finc
 **********************************************************/

extern int  __wrap_shmem_int_finc(int * a1, int a2) ;
extern int  __real_shmem_int_finc(int * a1, int a2)  {

  int retval;
  typedef int (*shmem_int_finc_t)(int * a1, int a2);
  static shmem_int_finc_t shmem_int_finc_handle = (shmem_int_finc_t)NULL;
  if (!shmem_int_finc_handle) {
    shmem_int_finc_handle = get_function_handle("shmem_int_finc");
  }

  retval  =  shmem_int_finc_handle ( a1,  a2);
  return retval;

}

extern int  shmem_int_finc(int * a1, int a2) {
   __wrap_shmem_int_finc(a1, a2);
}


/**********************************************************
   shmem_long_finc
 **********************************************************/

extern long  __wrap_shmem_long_finc(long * a1, int a2) ;
extern long  __real_shmem_long_finc(long * a1, int a2)  {

  long retval;
  typedef long (*shmem_long_finc_t)(long * a1, int a2);
  static shmem_long_finc_t shmem_long_finc_handle = (shmem_long_finc_t)NULL;
  if (!shmem_long_finc_handle) {
    shmem_long_finc_handle = get_function_handle("shmem_long_finc");
  }

  retval  =  shmem_long_finc_handle ( a1,  a2);
  return retval;

}

extern long  shmem_long_finc(long * a1, int a2) {
   __wrap_shmem_long_finc(a1, a2);
}


/**********************************************************
   shmem_longlong_finc
 **********************************************************/

extern long long  __wrap_shmem_longlong_finc(long long * a1, int a2) ;
extern long long  __real_shmem_longlong_finc(long long * a1, int a2)  {

  long long retval;
  typedef long long (*shmem_longlong_finc_t)(long long * a1, int a2);
  static shmem_longlong_finc_t shmem_longlong_finc_handle = (shmem_longlong_finc_t)NULL;
  if (!shmem_longlong_finc_handle) {
    shmem_longlong_finc_handle = get_function_handle("shmem_longlong_finc");
  }

  retval  =  shmem_longlong_finc_handle ( a1,  a2);
  return retval;

}

extern long long  shmem_longlong_finc(long long * a1, int a2) {
   __wrap_shmem_longlong_finc(a1, a2);
}


/**********************************************************
   shmem_short_finc_nb
 **********************************************************/

extern void  __wrap_shmem_short_finc_nb(short * a1, short * a2, int a3, void ** a4) ;
extern void  __real_shmem_short_finc_nb(short * a1, short * a2, int a3, void ** a4)  {

  typedef void (*shmem_short_finc_nb_t)(short * a1, short * a2, int a3, void ** a4);
  static shmem_short_finc_nb_t shmem_short_finc_nb_handle = (shmem_short_finc_nb_t)NULL;
  if (!shmem_short_finc_nb_handle) {
    shmem_short_finc_nb_handle = get_function_handle("shmem_short_finc_nb");
  }

  shmem_short_finc_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_short_finc_nb(short * a1, short * a2, int a3, void ** a4) {
   __wrap_shmem_short_finc_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_finc_nb
 **********************************************************/

extern void  __wrap_shmem_int_finc_nb(int * a1, int * a2, int a3, void ** a4) ;
extern void  __real_shmem_int_finc_nb(int * a1, int * a2, int a3, void ** a4)  {

  typedef void (*shmem_int_finc_nb_t)(int * a1, int * a2, int a3, void ** a4);
  static shmem_int_finc_nb_t shmem_int_finc_nb_handle = (shmem_int_finc_nb_t)NULL;
  if (!shmem_int_finc_nb_handle) {
    shmem_int_finc_nb_handle = get_function_handle("shmem_int_finc_nb");
  }

  shmem_int_finc_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_int_finc_nb(int * a1, int * a2, int a3, void ** a4) {
   __wrap_shmem_int_finc_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_finc_nb
 **********************************************************/

extern void  __wrap_shmem_long_finc_nb(long * a1, long * a2, int a3, void ** a4) ;
extern void  __real_shmem_long_finc_nb(long * a1, long * a2, int a3, void ** a4)  {

  typedef void (*shmem_long_finc_nb_t)(long * a1, long * a2, int a3, void ** a4);
  static shmem_long_finc_nb_t shmem_long_finc_nb_handle = (shmem_long_finc_nb_t)NULL;
  if (!shmem_long_finc_nb_handle) {
    shmem_long_finc_nb_handle = get_function_handle("shmem_long_finc_nb");
  }

  shmem_long_finc_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_long_finc_nb(long * a1, long * a2, int a3, void ** a4) {
   __wrap_shmem_long_finc_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_finc_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_finc_nb(long long * a1, long long * a2, int a3, void ** a4) ;
extern void  __real_shmem_longlong_finc_nb(long long * a1, long long * a2, int a3, void ** a4)  {

  typedef void (*shmem_longlong_finc_nb_t)(long long * a1, long long * a2, int a3, void ** a4);
  static shmem_longlong_finc_nb_t shmem_longlong_finc_nb_handle = (shmem_longlong_finc_nb_t)NULL;
  if (!shmem_longlong_finc_nb_handle) {
    shmem_longlong_finc_nb_handle = get_function_handle("shmem_longlong_finc_nb");
  }

  shmem_longlong_finc_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_finc_nb(long long * a1, long long * a2, int a3, void ** a4) {
   __wrap_shmem_longlong_finc_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_short_inc
 **********************************************************/

extern void  __wrap_shmem_short_inc(short * a1, int a2) ;
extern void  __real_shmem_short_inc(short * a1, int a2)  {

  typedef void (*shmem_short_inc_t)(short * a1, int a2);
  static shmem_short_inc_t shmem_short_inc_handle = (shmem_short_inc_t)NULL;
  if (!shmem_short_inc_handle) {
    shmem_short_inc_handle = get_function_handle("shmem_short_inc");
  }

  shmem_short_inc_handle ( a1,  a2);

}

extern void  shmem_short_inc(short * a1, int a2) {
   __wrap_shmem_short_inc(a1, a2);
}


/**********************************************************
   shmem_int_inc
 **********************************************************/

extern void  __wrap_shmem_int_inc(int * a1, int a2) ;
extern void  __real_shmem_int_inc(int * a1, int a2)  {

  typedef void (*shmem_int_inc_t)(int * a1, int a2);
  static shmem_int_inc_t shmem_int_inc_handle = (shmem_int_inc_t)NULL;
  if (!shmem_int_inc_handle) {
    shmem_int_inc_handle = get_function_handle("shmem_int_inc");
  }

  shmem_int_inc_handle ( a1,  a2);

}

extern void  shmem_int_inc(int * a1, int a2) {
   __wrap_shmem_int_inc(a1, a2);
}


/**********************************************************
   shmem_long_inc
 **********************************************************/

extern void  __wrap_shmem_long_inc(long * a1, int a2) ;
extern void  __real_shmem_long_inc(long * a1, int a2)  {

  typedef void (*shmem_long_inc_t)(long * a1, int a2);
  static shmem_long_inc_t shmem_long_inc_handle = (shmem_long_inc_t)NULL;
  if (!shmem_long_inc_handle) {
    shmem_long_inc_handle = get_function_handle("shmem_long_inc");
  }

  shmem_long_inc_handle ( a1,  a2);

}

extern void  shmem_long_inc(long * a1, int a2) {
   __wrap_shmem_long_inc(a1, a2);
}


/**********************************************************
   shmem_longlong_inc
 **********************************************************/

extern void  __wrap_shmem_longlong_inc(long long * a1, int a2) ;
extern void  __real_shmem_longlong_inc(long long * a1, int a2)  {

  typedef void (*shmem_longlong_inc_t)(long long * a1, int a2);
  static shmem_longlong_inc_t shmem_longlong_inc_handle = (shmem_longlong_inc_t)NULL;
  if (!shmem_longlong_inc_handle) {
    shmem_longlong_inc_handle = get_function_handle("shmem_longlong_inc");
  }

  shmem_longlong_inc_handle ( a1,  a2);

}

extern void  shmem_longlong_inc(long long * a1, int a2) {
   __wrap_shmem_longlong_inc(a1, a2);
}


/**********************************************************
   shmem_short_inc_nb
 **********************************************************/

extern void  __wrap_shmem_short_inc_nb(short * a1, int a2, void ** a3) ;
extern void  __real_shmem_short_inc_nb(short * a1, int a2, void ** a3)  {

  typedef void (*shmem_short_inc_nb_t)(short * a1, int a2, void ** a3);
  static shmem_short_inc_nb_t shmem_short_inc_nb_handle = (shmem_short_inc_nb_t)NULL;
  if (!shmem_short_inc_nb_handle) {
    shmem_short_inc_nb_handle = get_function_handle("shmem_short_inc_nb");
  }

  shmem_short_inc_nb_handle ( a1,  a2,  a3);

}

extern void  shmem_short_inc_nb(short * a1, int a2, void ** a3) {
   __wrap_shmem_short_inc_nb(a1, a2, a3);
}


/**********************************************************
   shmem_int_inc_nb
 **********************************************************/

extern void  __wrap_shmem_int_inc_nb(int * a1, int a2, void ** a3) ;
extern void  __real_shmem_int_inc_nb(int * a1, int a2, void ** a3)  {

  typedef void (*shmem_int_inc_nb_t)(int * a1, int a2, void ** a3);
  static shmem_int_inc_nb_t shmem_int_inc_nb_handle = (shmem_int_inc_nb_t)NULL;
  if (!shmem_int_inc_nb_handle) {
    shmem_int_inc_nb_handle = get_function_handle("shmem_int_inc_nb");
  }

  shmem_int_inc_nb_handle ( a1,  a2,  a3);

}

extern void  shmem_int_inc_nb(int * a1, int a2, void ** a3) {
   __wrap_shmem_int_inc_nb(a1, a2, a3);
}


/**********************************************************
   shmem_long_inc_nb
 **********************************************************/

extern void  __wrap_shmem_long_inc_nb(long * a1, int a2, void ** a3) ;
extern void  __real_shmem_long_inc_nb(long * a1, int a2, void ** a3)  {

  typedef void (*shmem_long_inc_nb_t)(long * a1, int a2, void ** a3);
  static shmem_long_inc_nb_t shmem_long_inc_nb_handle = (shmem_long_inc_nb_t)NULL;
  if (!shmem_long_inc_nb_handle) {
    shmem_long_inc_nb_handle = get_function_handle("shmem_long_inc_nb");
  }

  shmem_long_inc_nb_handle ( a1,  a2,  a3);

}

extern void  shmem_long_inc_nb(long * a1, int a2, void ** a3) {
   __wrap_shmem_long_inc_nb(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_inc_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_inc_nb(long long * a1, int a2, void ** a3) ;
extern void  __real_shmem_longlong_inc_nb(long long * a1, int a2, void ** a3)  {

  typedef void (*shmem_longlong_inc_nb_t)(long long * a1, int a2, void ** a3);
  static shmem_longlong_inc_nb_t shmem_longlong_inc_nb_handle = (shmem_longlong_inc_nb_t)NULL;
  if (!shmem_longlong_inc_nb_handle) {
    shmem_longlong_inc_nb_handle = get_function_handle("shmem_longlong_inc_nb");
  }

  shmem_longlong_inc_nb_handle ( a1,  a2,  a3);

}

extern void  shmem_longlong_inc_nb(long long * a1, int a2, void ** a3) {
   __wrap_shmem_longlong_inc_nb(a1, a2, a3);
}


/**********************************************************
   shmem_short_fadd
 **********************************************************/

extern short  __wrap_shmem_short_fadd(short * a1, short a2, int a3) ;
extern short  __real_shmem_short_fadd(short * a1, short a2, int a3)  {

  short retval;
  typedef short (*shmem_short_fadd_t)(short * a1, short a2, int a3);
  static shmem_short_fadd_t shmem_short_fadd_handle = (shmem_short_fadd_t)NULL;
  if (!shmem_short_fadd_handle) {
    shmem_short_fadd_handle = get_function_handle("shmem_short_fadd");
  }

  retval  =  shmem_short_fadd_handle ( a1,  a2,  a3);
  return retval;

}

extern short  shmem_short_fadd(short * a1, short a2, int a3) {
   __wrap_shmem_short_fadd(a1, a2, a3);
}


/**********************************************************
   shmem_int_fadd
 **********************************************************/

extern int  __wrap_shmem_int_fadd(int * a1, int a2, int a3) ;
extern int  __real_shmem_int_fadd(int * a1, int a2, int a3)  {

  int retval;
  typedef int (*shmem_int_fadd_t)(int * a1, int a2, int a3);
  static shmem_int_fadd_t shmem_int_fadd_handle = (shmem_int_fadd_t)NULL;
  if (!shmem_int_fadd_handle) {
    shmem_int_fadd_handle = get_function_handle("shmem_int_fadd");
  }

  retval  =  shmem_int_fadd_handle ( a1,  a2,  a3);
  return retval;

}

extern int  shmem_int_fadd(int * a1, int a2, int a3) {
   __wrap_shmem_int_fadd(a1, a2, a3);
}


/**********************************************************
   shmem_long_fadd
 **********************************************************/

extern long  __wrap_shmem_long_fadd(long * a1, long a2, int a3) ;
extern long  __real_shmem_long_fadd(long * a1, long a2, int a3)  {

  long retval;
  typedef long (*shmem_long_fadd_t)(long * a1, long a2, int a3);
  static shmem_long_fadd_t shmem_long_fadd_handle = (shmem_long_fadd_t)NULL;
  if (!shmem_long_fadd_handle) {
    shmem_long_fadd_handle = get_function_handle("shmem_long_fadd");
  }

  retval  =  shmem_long_fadd_handle ( a1,  a2,  a3);
  return retval;

}

extern long  shmem_long_fadd(long * a1, long a2, int a3) {
   __wrap_shmem_long_fadd(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_fadd
 **********************************************************/

extern long long  __wrap_shmem_longlong_fadd(long long * a1, long long a2, int a3) ;
extern long long  __real_shmem_longlong_fadd(long long * a1, long long a2, int a3)  {

  long long retval;
  typedef long long (*shmem_longlong_fadd_t)(long long * a1, long long a2, int a3);
  static shmem_longlong_fadd_t shmem_longlong_fadd_handle = (shmem_longlong_fadd_t)NULL;
  if (!shmem_longlong_fadd_handle) {
    shmem_longlong_fadd_handle = get_function_handle("shmem_longlong_fadd");
  }

  retval  =  shmem_longlong_fadd_handle ( a1,  a2,  a3);
  return retval;

}

extern long long  shmem_longlong_fadd(long long * a1, long long a2, int a3) {
   __wrap_shmem_longlong_fadd(a1, a2, a3);
}


/**********************************************************
   shmem_short_fadd_nb
 **********************************************************/

extern void  __wrap_shmem_short_fadd_nb(short * a1, short * a2, short a3, int a4, void ** a5) ;
extern void  __real_shmem_short_fadd_nb(short * a1, short * a2, short a3, int a4, void ** a5)  {

  typedef void (*shmem_short_fadd_nb_t)(short * a1, short * a2, short a3, int a4, void ** a5);
  static shmem_short_fadd_nb_t shmem_short_fadd_nb_handle = (shmem_short_fadd_nb_t)NULL;
  if (!shmem_short_fadd_nb_handle) {
    shmem_short_fadd_nb_handle = get_function_handle("shmem_short_fadd_nb");
  }

  shmem_short_fadd_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_short_fadd_nb(short * a1, short * a2, short a3, int a4, void ** a5) {
   __wrap_shmem_short_fadd_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_int_fadd_nb
 **********************************************************/

extern void  __wrap_shmem_int_fadd_nb(int * a1, int * a2, int a3, int a4, void ** a5) ;
extern void  __real_shmem_int_fadd_nb(int * a1, int * a2, int a3, int a4, void ** a5)  {

  typedef void (*shmem_int_fadd_nb_t)(int * a1, int * a2, int a3, int a4, void ** a5);
  static shmem_int_fadd_nb_t shmem_int_fadd_nb_handle = (shmem_int_fadd_nb_t)NULL;
  if (!shmem_int_fadd_nb_handle) {
    shmem_int_fadd_nb_handle = get_function_handle("shmem_int_fadd_nb");
  }

  shmem_int_fadd_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_int_fadd_nb(int * a1, int * a2, int a3, int a4, void ** a5) {
   __wrap_shmem_int_fadd_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_long_fadd_nb
 **********************************************************/

extern void  __wrap_shmem_long_fadd_nb(long * a1, long * a2, long a3, int a4, void ** a5) ;
extern void  __real_shmem_long_fadd_nb(long * a1, long * a2, long a3, int a4, void ** a5)  {

  typedef void (*shmem_long_fadd_nb_t)(long * a1, long * a2, long a3, int a4, void ** a5);
  static shmem_long_fadd_nb_t shmem_long_fadd_nb_handle = (shmem_long_fadd_nb_t)NULL;
  if (!shmem_long_fadd_nb_handle) {
    shmem_long_fadd_nb_handle = get_function_handle("shmem_long_fadd_nb");
  }

  shmem_long_fadd_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_long_fadd_nb(long * a1, long * a2, long a3, int a4, void ** a5) {
   __wrap_shmem_long_fadd_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_longlong_fadd_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_fadd_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) ;
extern void  __real_shmem_longlong_fadd_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5)  {

  typedef void (*shmem_longlong_fadd_nb_t)(long long * a1, long long * a2, long long a3, int a4, void ** a5);
  static shmem_longlong_fadd_nb_t shmem_longlong_fadd_nb_handle = (shmem_longlong_fadd_nb_t)NULL;
  if (!shmem_longlong_fadd_nb_handle) {
    shmem_longlong_fadd_nb_handle = get_function_handle("shmem_longlong_fadd_nb");
  }

  shmem_longlong_fadd_nb_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_longlong_fadd_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) {
   __wrap_shmem_longlong_fadd_nb(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_short_add
 **********************************************************/

extern void  __wrap_shmem_short_add(short * a1, short a2, int a3) ;
extern void  __real_shmem_short_add(short * a1, short a2, int a3)  {

  typedef void (*shmem_short_add_t)(short * a1, short a2, int a3);
  static shmem_short_add_t shmem_short_add_handle = (shmem_short_add_t)NULL;
  if (!shmem_short_add_handle) {
    shmem_short_add_handle = get_function_handle("shmem_short_add");
  }

  shmem_short_add_handle ( a1,  a2,  a3);

}

extern void  shmem_short_add(short * a1, short a2, int a3) {
   __wrap_shmem_short_add(a1, a2, a3);
}


/**********************************************************
   shmem_int_add
 **********************************************************/

extern void  __wrap_shmem_int_add(int * a1, int a2, int a3) ;
extern void  __real_shmem_int_add(int * a1, int a2, int a3)  {

  typedef void (*shmem_int_add_t)(int * a1, int a2, int a3);
  static shmem_int_add_t shmem_int_add_handle = (shmem_int_add_t)NULL;
  if (!shmem_int_add_handle) {
    shmem_int_add_handle = get_function_handle("shmem_int_add");
  }

  shmem_int_add_handle ( a1,  a2,  a3);

}

extern void  shmem_int_add(int * a1, int a2, int a3) {
   __wrap_shmem_int_add(a1, a2, a3);
}


/**********************************************************
   shmem_long_add
 **********************************************************/

extern void  __wrap_shmem_long_add(long * a1, long a2, int a3) ;
extern void  __real_shmem_long_add(long * a1, long a2, int a3)  {

  typedef void (*shmem_long_add_t)(long * a1, long a2, int a3);
  static shmem_long_add_t shmem_long_add_handle = (shmem_long_add_t)NULL;
  if (!shmem_long_add_handle) {
    shmem_long_add_handle = get_function_handle("shmem_long_add");
  }

  shmem_long_add_handle ( a1,  a2,  a3);

}

extern void  shmem_long_add(long * a1, long a2, int a3) {
   __wrap_shmem_long_add(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_add
 **********************************************************/

extern void  __wrap_shmem_longlong_add(long long * a1, long long a2, int a3) ;
extern void  __real_shmem_longlong_add(long long * a1, long long a2, int a3)  {

  typedef void (*shmem_longlong_add_t)(long long * a1, long long a2, int a3);
  static shmem_longlong_add_t shmem_longlong_add_handle = (shmem_longlong_add_t)NULL;
  if (!shmem_longlong_add_handle) {
    shmem_longlong_add_handle = get_function_handle("shmem_longlong_add");
  }

  shmem_longlong_add_handle ( a1,  a2,  a3);

}

extern void  shmem_longlong_add(long long * a1, long long a2, int a3) {
   __wrap_shmem_longlong_add(a1, a2, a3);
}


/**********************************************************
   shmem_short_add_nb
 **********************************************************/

extern void  __wrap_shmem_short_add_nb(short * a1, short a2, int a3, void ** a4) ;
extern void  __real_shmem_short_add_nb(short * a1, short a2, int a3, void ** a4)  {

  typedef void (*shmem_short_add_nb_t)(short * a1, short a2, int a3, void ** a4);
  static shmem_short_add_nb_t shmem_short_add_nb_handle = (shmem_short_add_nb_t)NULL;
  if (!shmem_short_add_nb_handle) {
    shmem_short_add_nb_handle = get_function_handle("shmem_short_add_nb");
  }

  shmem_short_add_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_short_add_nb(short * a1, short a2, int a3, void ** a4) {
   __wrap_shmem_short_add_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_add_nb
 **********************************************************/

extern void  __wrap_shmem_int_add_nb(int * a1, int a2, int a3, void ** a4) ;
extern void  __real_shmem_int_add_nb(int * a1, int a2, int a3, void ** a4)  {

  typedef void (*shmem_int_add_nb_t)(int * a1, int a2, int a3, void ** a4);
  static shmem_int_add_nb_t shmem_int_add_nb_handle = (shmem_int_add_nb_t)NULL;
  if (!shmem_int_add_nb_handle) {
    shmem_int_add_nb_handle = get_function_handle("shmem_int_add_nb");
  }

  shmem_int_add_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_int_add_nb(int * a1, int a2, int a3, void ** a4) {
   __wrap_shmem_int_add_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_long_add_nb
 **********************************************************/

extern void  __wrap_shmem_long_add_nb(long * a1, long a2, int a3, void ** a4) ;
extern void  __real_shmem_long_add_nb(long * a1, long a2, int a3, void ** a4)  {

  typedef void (*shmem_long_add_nb_t)(long * a1, long a2, int a3, void ** a4);
  static shmem_long_add_nb_t shmem_long_add_nb_handle = (shmem_long_add_nb_t)NULL;
  if (!shmem_long_add_nb_handle) {
    shmem_long_add_nb_handle = get_function_handle("shmem_long_add_nb");
  }

  shmem_long_add_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_long_add_nb(long * a1, long a2, int a3, void ** a4) {
   __wrap_shmem_long_add_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_longlong_add_nb
 **********************************************************/

extern void  __wrap_shmem_longlong_add_nb(long long * a1, long long a2, int a3, void ** a4) ;
extern void  __real_shmem_longlong_add_nb(long long * a1, long long a2, int a3, void ** a4)  {

  typedef void (*shmem_longlong_add_nb_t)(long long * a1, long long a2, int a3, void ** a4);
  static shmem_longlong_add_nb_t shmem_longlong_add_nb_handle = (shmem_longlong_add_nb_t)NULL;
  if (!shmem_longlong_add_nb_handle) {
    shmem_longlong_add_nb_handle = get_function_handle("shmem_longlong_add_nb");
  }

  shmem_longlong_add_nb_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_longlong_add_nb(long long * a1, long long a2, int a3, void ** a4) {
   __wrap_shmem_longlong_add_nb(a1, a2, a3, a4);
}


/**********************************************************
   shmem_int_fetch
 **********************************************************/

extern int  __wrap_shmem_int_fetch(const int * a1, int a2) ;
extern int  __real_shmem_int_fetch(const int * a1, int a2)  {

  int retval;
  typedef int (*shmem_int_fetch_t)(const int * a1, int a2);
  static shmem_int_fetch_t shmem_int_fetch_handle = (shmem_int_fetch_t)NULL;
  if (!shmem_int_fetch_handle) {
    shmem_int_fetch_handle = get_function_handle("shmem_int_fetch");
  }

  retval  =  shmem_int_fetch_handle ( a1,  a2);
  return retval;

}

extern int  shmem_int_fetch(const int * a1, int a2) {
   __wrap_shmem_int_fetch(a1, a2);
}


/**********************************************************
   shmem_long_fetch
 **********************************************************/

extern long  __wrap_shmem_long_fetch(const long * a1, int a2) ;
extern long  __real_shmem_long_fetch(const long * a1, int a2)  {

  long retval;
  typedef long (*shmem_long_fetch_t)(const long * a1, int a2);
  static shmem_long_fetch_t shmem_long_fetch_handle = (shmem_long_fetch_t)NULL;
  if (!shmem_long_fetch_handle) {
    shmem_long_fetch_handle = get_function_handle("shmem_long_fetch");
  }

  retval  =  shmem_long_fetch_handle ( a1,  a2);
  return retval;

}

extern long  shmem_long_fetch(const long * a1, int a2) {
   __wrap_shmem_long_fetch(a1, a2);
}


/**********************************************************
   shmem_float_fetch
 **********************************************************/

extern float  __wrap_shmem_float_fetch(const float * a1, int a2) ;
extern float  __real_shmem_float_fetch(const float * a1, int a2)  {

  float retval;
  typedef float (*shmem_float_fetch_t)(const float * a1, int a2);
  static shmem_float_fetch_t shmem_float_fetch_handle = (shmem_float_fetch_t)NULL;
  if (!shmem_float_fetch_handle) {
    shmem_float_fetch_handle = get_function_handle("shmem_float_fetch");
  }

  retval  =  shmem_float_fetch_handle ( a1,  a2);
  return retval;

}

extern float  shmem_float_fetch(const float * a1, int a2) {
   __wrap_shmem_float_fetch(a1, a2);
}


/**********************************************************
   shmem_double_fetch
 **********************************************************/

extern double  __wrap_shmem_double_fetch(const double * a1, int a2) ;
extern double  __real_shmem_double_fetch(const double * a1, int a2)  {

  double retval;
  typedef double (*shmem_double_fetch_t)(const double * a1, int a2);
  static shmem_double_fetch_t shmem_double_fetch_handle = (shmem_double_fetch_t)NULL;
  if (!shmem_double_fetch_handle) {
    shmem_double_fetch_handle = get_function_handle("shmem_double_fetch");
  }

  retval  =  shmem_double_fetch_handle ( a1,  a2);
  return retval;

}

extern double  shmem_double_fetch(const double * a1, int a2) {
   __wrap_shmem_double_fetch(a1, a2);
}


/**********************************************************
   shmem_longlong_fetch
 **********************************************************/

extern long long  __wrap_shmem_longlong_fetch(const long long * a1, int a2) ;
extern long long  __real_shmem_longlong_fetch(const long long * a1, int a2)  {

  long long retval;
  typedef long long (*shmem_longlong_fetch_t)(const long long * a1, int a2);
  static shmem_longlong_fetch_t shmem_longlong_fetch_handle = (shmem_longlong_fetch_t)NULL;
  if (!shmem_longlong_fetch_handle) {
    shmem_longlong_fetch_handle = get_function_handle("shmem_longlong_fetch");
  }

  retval  =  shmem_longlong_fetch_handle ( a1,  a2);
  return retval;

}

extern long long  shmem_longlong_fetch(const long long * a1, int a2) {
   __wrap_shmem_longlong_fetch(a1, a2);
}


/**********************************************************
   shmem_int_set
 **********************************************************/

extern void  __wrap_shmem_int_set(int * a1, int a2, int a3) ;
extern void  __real_shmem_int_set(int * a1, int a2, int a3)  {

  typedef void (*shmem_int_set_t)(int * a1, int a2, int a3);
  static shmem_int_set_t shmem_int_set_handle = (shmem_int_set_t)NULL;
  if (!shmem_int_set_handle) {
    shmem_int_set_handle = get_function_handle("shmem_int_set");
  }

  shmem_int_set_handle ( a1,  a2,  a3);

}

extern void  shmem_int_set(int * a1, int a2, int a3) {
   __wrap_shmem_int_set(a1, a2, a3);
}


/**********************************************************
   shmem_long_set
 **********************************************************/

extern void  __wrap_shmem_long_set(long * a1, long a2, int a3) ;
extern void  __real_shmem_long_set(long * a1, long a2, int a3)  {

  typedef void (*shmem_long_set_t)(long * a1, long a2, int a3);
  static shmem_long_set_t shmem_long_set_handle = (shmem_long_set_t)NULL;
  if (!shmem_long_set_handle) {
    shmem_long_set_handle = get_function_handle("shmem_long_set");
  }

  shmem_long_set_handle ( a1,  a2,  a3);

}

extern void  shmem_long_set(long * a1, long a2, int a3) {
   __wrap_shmem_long_set(a1, a2, a3);
}


/**********************************************************
   shmem_float_set
 **********************************************************/

extern void  __wrap_shmem_float_set(float * a1, float a2, int a3) ;
extern void  __real_shmem_float_set(float * a1, float a2, int a3)  {

  typedef void (*shmem_float_set_t)(float * a1, float a2, int a3);
  static shmem_float_set_t shmem_float_set_handle = (shmem_float_set_t)NULL;
  if (!shmem_float_set_handle) {
    shmem_float_set_handle = get_function_handle("shmem_float_set");
  }

  shmem_float_set_handle ( a1,  a2,  a3);

}

extern void  shmem_float_set(float * a1, float a2, int a3) {
   __wrap_shmem_float_set(a1, a2, a3);
}


/**********************************************************
   shmem_double_set
 **********************************************************/

extern void  __wrap_shmem_double_set(double * a1, double a2, int a3) ;
extern void  __real_shmem_double_set(double * a1, double a2, int a3)  {

  typedef void (*shmem_double_set_t)(double * a1, double a2, int a3);
  static shmem_double_set_t shmem_double_set_handle = (shmem_double_set_t)NULL;
  if (!shmem_double_set_handle) {
    shmem_double_set_handle = get_function_handle("shmem_double_set");
  }

  shmem_double_set_handle ( a1,  a2,  a3);

}

extern void  shmem_double_set(double * a1, double a2, int a3) {
   __wrap_shmem_double_set(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_set
 **********************************************************/

extern void  __wrap_shmem_longlong_set(long long * a1, long long a2, int a3) ;
extern void  __real_shmem_longlong_set(long long * a1, long long a2, int a3)  {

  typedef void (*shmem_longlong_set_t)(long long * a1, long long a2, int a3);
  static shmem_longlong_set_t shmem_longlong_set_handle = (shmem_longlong_set_t)NULL;
  if (!shmem_longlong_set_handle) {
    shmem_longlong_set_handle = get_function_handle("shmem_longlong_set");
  }

  shmem_longlong_set_handle ( a1,  a2,  a3);

}

extern void  shmem_longlong_set(long long * a1, long long a2, int a3) {
   __wrap_shmem_longlong_set(a1, a2, a3);
}


/**********************************************************
   shmem_barrier_all
 **********************************************************/

extern void  __wrap_shmem_barrier_all() ;
extern void  __real_shmem_barrier_all()  {

  typedef void (*shmem_barrier_all_t)();
  static shmem_barrier_all_t shmem_barrier_all_handle = (shmem_barrier_all_t)NULL;
  if (!shmem_barrier_all_handle) {
    shmem_barrier_all_handle = get_function_handle("shmem_barrier_all");
  }

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
  static shmem_barrier_t shmem_barrier_handle = (shmem_barrier_t)NULL;
  if (!shmem_barrier_handle) {
    shmem_barrier_handle = get_function_handle("shmem_barrier");
  }

  shmem_barrier_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_barrier(int a1, int a2, int a3, long * a4) {
   __wrap_shmem_barrier(a1, a2, a3, a4);
}


/**********************************************************
   shmem_team_barrier
 **********************************************************/

extern void  __wrap_shmem_team_barrier(shmem_team_t a1, long * a2) ;
extern void  __real_shmem_team_barrier(shmem_team_t a1, long * a2)  {

  typedef void (*shmem_team_barrier_t)(shmem_team_t a1, long * a2);
  static shmem_team_barrier_t shmem_team_barrier_handle = (shmem_team_barrier_t)NULL;
  if (!shmem_team_barrier_handle) {
    shmem_team_barrier_handle = get_function_handle("shmem_team_barrier");
  }

  shmem_team_barrier_handle ( a1,  a2);

}

extern void  shmem_team_barrier(shmem_team_t a1, long * a2) {
   __wrap_shmem_team_barrier(a1, a2);
}


/**********************************************************
   shmem_fence
 **********************************************************/

extern void  __wrap_shmem_fence() ;
extern void  __real_shmem_fence()  {

  typedef void (*shmem_fence_t)();
  static shmem_fence_t shmem_fence_handle = (shmem_fence_t)NULL;
  if (!shmem_fence_handle) {
    shmem_fence_handle = get_function_handle("shmem_fence");
  }

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
  static shmem_quiet_t shmem_quiet_handle = (shmem_quiet_t)NULL;
  if (!shmem_quiet_handle) {
    shmem_quiet_handle = get_function_handle("shmem_quiet");
  }

  shmem_quiet_handle ();

}

extern void  shmem_quiet() {
   __wrap_shmem_quiet();
}


/**********************************************************
   shmem_set_lock
 **********************************************************/

extern void  __wrap_shmem_set_lock(long * a1) ;
extern void  __real_shmem_set_lock(long * a1)  {

  typedef void (*shmem_set_lock_t)(long * a1);
  static shmem_set_lock_t shmem_set_lock_handle = (shmem_set_lock_t)NULL;
  if (!shmem_set_lock_handle) {
    shmem_set_lock_handle = get_function_handle("shmem_set_lock");
  }

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
  static shmem_clear_lock_t shmem_clear_lock_handle = (shmem_clear_lock_t)NULL;
  if (!shmem_clear_lock_handle) {
    shmem_clear_lock_handle = get_function_handle("shmem_clear_lock");
  }

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
  static shmem_test_lock_t shmem_test_lock_handle = (shmem_test_lock_t)NULL;
  if (!shmem_test_lock_handle) {
    shmem_test_lock_handle = get_function_handle("shmem_test_lock");
  }

  retval  =  shmem_test_lock_handle ( a1);
  return retval;

}

extern int  shmem_test_lock(long * a1) {
   __wrap_shmem_test_lock(a1);
}


/**********************************************************
   shmem_clear_event
 **********************************************************/

extern void  __wrap_shmem_clear_event(long * a1) ;
extern void  __real_shmem_clear_event(long * a1)  {

  typedef void (*shmem_clear_event_t)(long * a1);
  static shmem_clear_event_t shmem_clear_event_handle = (shmem_clear_event_t)NULL;
  if (!shmem_clear_event_handle) {
    shmem_clear_event_handle = get_function_handle("shmem_clear_event");
  }

  shmem_clear_event_handle ( a1);

}

extern void  shmem_clear_event(long * a1) {
   __wrap_shmem_clear_event(a1);
}


/**********************************************************
   shmem_set_event
 **********************************************************/

extern void  __wrap_shmem_set_event(long * a1) ;
extern void  __real_shmem_set_event(long * a1)  {

  typedef void (*shmem_set_event_t)(long * a1);
  static shmem_set_event_t shmem_set_event_handle = (shmem_set_event_t)NULL;
  if (!shmem_set_event_handle) {
    shmem_set_event_handle = get_function_handle("shmem_set_event");
  }

  shmem_set_event_handle ( a1);

}

extern void  shmem_set_event(long * a1) {
   __wrap_shmem_set_event(a1);
}


/**********************************************************
   shmem_test_event
 **********************************************************/

extern int  __wrap_shmem_test_event(long * a1) ;
extern int  __real_shmem_test_event(long * a1)  {

  int retval;
  typedef int (*shmem_test_event_t)(long * a1);
  static shmem_test_event_t shmem_test_event_handle = (shmem_test_event_t)NULL;
  if (!shmem_test_event_handle) {
    shmem_test_event_handle = get_function_handle("shmem_test_event");
  }

  retval  =  shmem_test_event_handle ( a1);
  return retval;

}

extern int  shmem_test_event(long * a1) {
   __wrap_shmem_test_event(a1);
}


/**********************************************************
   shmem_wait_event
 **********************************************************/

extern void  __wrap_shmem_wait_event(long * a1) ;
extern void  __real_shmem_wait_event(long * a1)  {

  typedef void (*shmem_wait_event_t)(long * a1);
  static shmem_wait_event_t shmem_wait_event_handle = (shmem_wait_event_t)NULL;
  if (!shmem_wait_event_handle) {
    shmem_wait_event_handle = get_function_handle("shmem_wait_event");
  }

  shmem_wait_event_handle ( a1);

}

extern void  shmem_wait_event(long * a1) {
   __wrap_shmem_wait_event(a1);
}


/**********************************************************
   shmem_short_wait
 **********************************************************/

extern void  __wrap_shmem_short_wait(short * a1, short a2) ;
extern void  __real_shmem_short_wait(short * a1, short a2)  {

  typedef void (*shmem_short_wait_t)(short * a1, short a2);
  static shmem_short_wait_t shmem_short_wait_handle = (shmem_short_wait_t)NULL;
  if (!shmem_short_wait_handle) {
    shmem_short_wait_handle = get_function_handle("shmem_short_wait");
  }

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
  static shmem_int_wait_t shmem_int_wait_handle = (shmem_int_wait_t)NULL;
  if (!shmem_int_wait_handle) {
    shmem_int_wait_handle = get_function_handle("shmem_int_wait");
  }

  shmem_int_wait_handle ( a1,  a2);

}

extern void  shmem_int_wait(int * a1, int a2) {
   __wrap_shmem_int_wait(a1, a2);
}


/**********************************************************
   shmem_long_wait
 **********************************************************/

extern void  __wrap_shmem_long_wait(long * a1, long a2) ;
extern void  __real_shmem_long_wait(long * a1, long a2)  {

  typedef void (*shmem_long_wait_t)(long * a1, long a2);
  static shmem_long_wait_t shmem_long_wait_handle = (shmem_long_wait_t)NULL;
  if (!shmem_long_wait_handle) {
    shmem_long_wait_handle = get_function_handle("shmem_long_wait");
  }

  shmem_long_wait_handle ( a1,  a2);

}

extern void  shmem_long_wait(long * a1, long a2) {
   __wrap_shmem_long_wait(a1, a2);
}


/**********************************************************
   shmem_longlong_wait
 **********************************************************/

extern void  __wrap_shmem_longlong_wait(long long * a1, long long a2) ;
extern void  __real_shmem_longlong_wait(long long * a1, long long a2)  {

  typedef void (*shmem_longlong_wait_t)(long long * a1, long long a2);
  static shmem_longlong_wait_t shmem_longlong_wait_handle = (shmem_longlong_wait_t)NULL;
  if (!shmem_longlong_wait_handle) {
    shmem_longlong_wait_handle = get_function_handle("shmem_longlong_wait");
  }

  shmem_longlong_wait_handle ( a1,  a2);

}

extern void  shmem_longlong_wait(long long * a1, long long a2) {
   __wrap_shmem_longlong_wait(a1, a2);
}


/**********************************************************
   shmem_short_wait_until
 **********************************************************/

extern void  __wrap_shmem_short_wait_until(short * a1, int a2, short a3) ;
extern void  __real_shmem_short_wait_until(short * a1, int a2, short a3)  {

  typedef void (*shmem_short_wait_until_t)(short * a1, int a2, short a3);
  static shmem_short_wait_until_t shmem_short_wait_until_handle = (shmem_short_wait_until_t)NULL;
  if (!shmem_short_wait_until_handle) {
    shmem_short_wait_until_handle = get_function_handle("shmem_short_wait_until");
  }

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
  static shmem_int_wait_until_t shmem_int_wait_until_handle = (shmem_int_wait_until_t)NULL;
  if (!shmem_int_wait_until_handle) {
    shmem_int_wait_until_handle = get_function_handle("shmem_int_wait_until");
  }

  shmem_int_wait_until_handle ( a1,  a2,  a3);

}

extern void  shmem_int_wait_until(int * a1, int a2, int a3) {
   __wrap_shmem_int_wait_until(a1, a2, a3);
}


/**********************************************************
   shmem_long_wait_until
 **********************************************************/

extern void  __wrap_shmem_long_wait_until(long * a1, int a2, long a3) ;
extern void  __real_shmem_long_wait_until(long * a1, int a2, long a3)  {

  typedef void (*shmem_long_wait_until_t)(long * a1, int a2, long a3);
  static shmem_long_wait_until_t shmem_long_wait_until_handle = (shmem_long_wait_until_t)NULL;
  if (!shmem_long_wait_until_handle) {
    shmem_long_wait_until_handle = get_function_handle("shmem_long_wait_until");
  }

  shmem_long_wait_until_handle ( a1,  a2,  a3);

}

extern void  shmem_long_wait_until(long * a1, int a2, long a3) {
   __wrap_shmem_long_wait_until(a1, a2, a3);
}


/**********************************************************
   shmem_longlong_wait_until
 **********************************************************/

extern void  __wrap_shmem_longlong_wait_until(long long * a1, int a2, long long a3) ;
extern void  __real_shmem_longlong_wait_until(long long * a1, int a2, long long a3)  {

  typedef void (*shmem_longlong_wait_until_t)(long long * a1, int a2, long long a3);
  static shmem_longlong_wait_until_t shmem_longlong_wait_until_handle = (shmem_longlong_wait_until_t)NULL;
  if (!shmem_longlong_wait_until_handle) {
    shmem_longlong_wait_until_handle = get_function_handle("shmem_longlong_wait_until");
  }

  shmem_longlong_wait_until_handle ( a1,  a2,  a3);

}

extern void  shmem_longlong_wait_until(long long * a1, int a2, long long a3) {
   __wrap_shmem_longlong_wait_until(a1, a2, a3);
}


/**********************************************************
   shmem_short_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_sum_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  static shmem_short_sum_to_all_t shmem_short_sum_to_all_handle = (shmem_short_sum_to_all_t)NULL;
  if (!shmem_short_sum_to_all_handle) {
    shmem_short_sum_to_all_handle = get_function_handle("shmem_short_sum_to_all");
  }

  shmem_short_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_max_to_all
 **********************************************************/

extern void  __wrap_shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_max_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  static shmem_short_max_to_all_t shmem_short_max_to_all_handle = (shmem_short_max_to_all_t)NULL;
  if (!shmem_short_max_to_all_handle) {
    shmem_short_max_to_all_handle = get_function_handle("shmem_short_max_to_all");
  }

  shmem_short_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_min_to_all
 **********************************************************/

extern void  __wrap_shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_min_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  static shmem_short_min_to_all_t shmem_short_min_to_all_handle = (shmem_short_min_to_all_t)NULL;
  if (!shmem_short_min_to_all_handle) {
    shmem_short_min_to_all_handle = get_function_handle("shmem_short_min_to_all");
  }

  shmem_short_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_prod_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  static shmem_short_prod_to_all_t shmem_short_prod_to_all_handle = (shmem_short_prod_to_all_t)NULL;
  if (!shmem_short_prod_to_all_handle) {
    shmem_short_prod_to_all_handle = get_function_handle("shmem_short_prod_to_all");
  }

  shmem_short_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_and_to_all
 **********************************************************/

extern void  __wrap_shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_and_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  static shmem_short_and_to_all_t shmem_short_and_to_all_handle = (shmem_short_and_to_all_t)NULL;
  if (!shmem_short_and_to_all_handle) {
    shmem_short_and_to_all_handle = get_function_handle("shmem_short_and_to_all");
  }

  shmem_short_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_or_to_all
 **********************************************************/

extern void  __wrap_shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_or_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  static shmem_short_or_to_all_t shmem_short_or_to_all_handle = (shmem_short_or_to_all_t)NULL;
  if (!shmem_short_or_to_all_handle) {
    shmem_short_or_to_all_handle = get_function_handle("shmem_short_or_to_all");
  }

  shmem_short_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_short_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __real_shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  typedef void (*shmem_short_xor_to_all_t)(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8);
  static shmem_short_xor_to_all_t shmem_short_xor_to_all_handle = (shmem_short_xor_to_all_t)NULL;
  if (!shmem_short_xor_to_all_handle) {
    shmem_short_xor_to_all_handle = get_function_handle("shmem_short_xor_to_all");
  }

  shmem_short_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {
   __wrap_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_sum_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  static shmem_int_sum_to_all_t shmem_int_sum_to_all_handle = (shmem_int_sum_to_all_t)NULL;
  if (!shmem_int_sum_to_all_handle) {
    shmem_int_sum_to_all_handle = get_function_handle("shmem_int_sum_to_all");
  }

  shmem_int_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_max_to_all
 **********************************************************/

extern void  __wrap_shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_max_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  static shmem_int_max_to_all_t shmem_int_max_to_all_handle = (shmem_int_max_to_all_t)NULL;
  if (!shmem_int_max_to_all_handle) {
    shmem_int_max_to_all_handle = get_function_handle("shmem_int_max_to_all");
  }

  shmem_int_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_min_to_all
 **********************************************************/

extern void  __wrap_shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_min_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  static shmem_int_min_to_all_t shmem_int_min_to_all_handle = (shmem_int_min_to_all_t)NULL;
  if (!shmem_int_min_to_all_handle) {
    shmem_int_min_to_all_handle = get_function_handle("shmem_int_min_to_all");
  }

  shmem_int_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_prod_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  static shmem_int_prod_to_all_t shmem_int_prod_to_all_handle = (shmem_int_prod_to_all_t)NULL;
  if (!shmem_int_prod_to_all_handle) {
    shmem_int_prod_to_all_handle = get_function_handle("shmem_int_prod_to_all");
  }

  shmem_int_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_and_to_all
 **********************************************************/

extern void  __wrap_shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_and_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  static shmem_int_and_to_all_t shmem_int_and_to_all_handle = (shmem_int_and_to_all_t)NULL;
  if (!shmem_int_and_to_all_handle) {
    shmem_int_and_to_all_handle = get_function_handle("shmem_int_and_to_all");
  }

  shmem_int_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_or_to_all
 **********************************************************/

extern void  __wrap_shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_or_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  static shmem_int_or_to_all_t shmem_int_or_to_all_handle = (shmem_int_or_to_all_t)NULL;
  if (!shmem_int_or_to_all_handle) {
    shmem_int_or_to_all_handle = get_function_handle("shmem_int_or_to_all");
  }

  shmem_int_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_int_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __real_shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  typedef void (*shmem_int_xor_to_all_t)(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8);
  static shmem_int_xor_to_all_t shmem_int_xor_to_all_handle = (shmem_int_xor_to_all_t)NULL;
  if (!shmem_int_xor_to_all_handle) {
    shmem_int_xor_to_all_handle = get_function_handle("shmem_int_xor_to_all");
  }

  shmem_int_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {
   __wrap_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_sum_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  static shmem_long_sum_to_all_t shmem_long_sum_to_all_handle = (shmem_long_sum_to_all_t)NULL;
  if (!shmem_long_sum_to_all_handle) {
    shmem_long_sum_to_all_handle = get_function_handle("shmem_long_sum_to_all");
  }

  shmem_long_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_max_to_all
 **********************************************************/

extern void  __wrap_shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_max_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  static shmem_long_max_to_all_t shmem_long_max_to_all_handle = (shmem_long_max_to_all_t)NULL;
  if (!shmem_long_max_to_all_handle) {
    shmem_long_max_to_all_handle = get_function_handle("shmem_long_max_to_all");
  }

  shmem_long_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_min_to_all
 **********************************************************/

extern void  __wrap_shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_min_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  static shmem_long_min_to_all_t shmem_long_min_to_all_handle = (shmem_long_min_to_all_t)NULL;
  if (!shmem_long_min_to_all_handle) {
    shmem_long_min_to_all_handle = get_function_handle("shmem_long_min_to_all");
  }

  shmem_long_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_prod_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  static shmem_long_prod_to_all_t shmem_long_prod_to_all_handle = (shmem_long_prod_to_all_t)NULL;
  if (!shmem_long_prod_to_all_handle) {
    shmem_long_prod_to_all_handle = get_function_handle("shmem_long_prod_to_all");
  }

  shmem_long_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_and_to_all
 **********************************************************/

extern void  __wrap_shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_and_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  static shmem_long_and_to_all_t shmem_long_and_to_all_handle = (shmem_long_and_to_all_t)NULL;
  if (!shmem_long_and_to_all_handle) {
    shmem_long_and_to_all_handle = get_function_handle("shmem_long_and_to_all");
  }

  shmem_long_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_or_to_all
 **********************************************************/

extern void  __wrap_shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_or_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  static shmem_long_or_to_all_t shmem_long_or_to_all_handle = (shmem_long_or_to_all_t)NULL;
  if (!shmem_long_or_to_all_handle) {
    shmem_long_or_to_all_handle = get_function_handle("shmem_long_or_to_all");
  }

  shmem_long_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_long_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __real_shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  typedef void (*shmem_long_xor_to_all_t)(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8);
  static shmem_long_xor_to_all_t shmem_long_xor_to_all_handle = (shmem_long_xor_to_all_t)NULL;
  if (!shmem_long_xor_to_all_handle) {
    shmem_long_xor_to_all_handle = get_function_handle("shmem_long_xor_to_all");
  }

  shmem_long_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {
   __wrap_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_sum_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  static shmem_longlong_sum_to_all_t shmem_longlong_sum_to_all_handle = (shmem_longlong_sum_to_all_t)NULL;
  if (!shmem_longlong_sum_to_all_handle) {
    shmem_longlong_sum_to_all_handle = get_function_handle("shmem_longlong_sum_to_all");
  }

  shmem_longlong_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_max_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_max_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  static shmem_longlong_max_to_all_t shmem_longlong_max_to_all_handle = (shmem_longlong_max_to_all_t)NULL;
  if (!shmem_longlong_max_to_all_handle) {
    shmem_longlong_max_to_all_handle = get_function_handle("shmem_longlong_max_to_all");
  }

  shmem_longlong_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_min_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_min_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  static shmem_longlong_min_to_all_t shmem_longlong_min_to_all_handle = (shmem_longlong_min_to_all_t)NULL;
  if (!shmem_longlong_min_to_all_handle) {
    shmem_longlong_min_to_all_handle = get_function_handle("shmem_longlong_min_to_all");
  }

  shmem_longlong_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_prod_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  static shmem_longlong_prod_to_all_t shmem_longlong_prod_to_all_handle = (shmem_longlong_prod_to_all_t)NULL;
  if (!shmem_longlong_prod_to_all_handle) {
    shmem_longlong_prod_to_all_handle = get_function_handle("shmem_longlong_prod_to_all");
  }

  shmem_longlong_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_and_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_and_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  static shmem_longlong_and_to_all_t shmem_longlong_and_to_all_handle = (shmem_longlong_and_to_all_t)NULL;
  if (!shmem_longlong_and_to_all_handle) {
    shmem_longlong_and_to_all_handle = get_function_handle("shmem_longlong_and_to_all");
  }

  shmem_longlong_and_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_or_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_or_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  static shmem_longlong_or_to_all_t shmem_longlong_or_to_all_handle = (shmem_longlong_or_to_all_t)NULL;
  if (!shmem_longlong_or_to_all_handle) {
    shmem_longlong_or_to_all_handle = get_function_handle("shmem_longlong_or_to_all");
  }

  shmem_longlong_or_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_longlong_xor_to_all
 **********************************************************/

extern void  __wrap_shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __real_shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  typedef void (*shmem_longlong_xor_to_all_t)(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8);
  static shmem_longlong_xor_to_all_t shmem_longlong_xor_to_all_handle = (shmem_longlong_xor_to_all_t)NULL;
  if (!shmem_longlong_xor_to_all_handle) {
    shmem_longlong_xor_to_all_handle = get_function_handle("shmem_longlong_xor_to_all");
  }

  shmem_longlong_xor_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {
   __wrap_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_sum_to_all_t)(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8);
  static shmem_float_sum_to_all_t shmem_float_sum_to_all_handle = (shmem_float_sum_to_all_t)NULL;
  if (!shmem_float_sum_to_all_handle) {
    shmem_float_sum_to_all_handle = get_function_handle("shmem_float_sum_to_all");
  }

  shmem_float_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_max_to_all
 **********************************************************/

extern void  __wrap_shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_max_to_all_t)(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8);
  static shmem_float_max_to_all_t shmem_float_max_to_all_handle = (shmem_float_max_to_all_t)NULL;
  if (!shmem_float_max_to_all_handle) {
    shmem_float_max_to_all_handle = get_function_handle("shmem_float_max_to_all");
  }

  shmem_float_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_min_to_all
 **********************************************************/

extern void  __wrap_shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_min_to_all_t)(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8);
  static shmem_float_min_to_all_t shmem_float_min_to_all_handle = (shmem_float_min_to_all_t)NULL;
  if (!shmem_float_min_to_all_handle) {
    shmem_float_min_to_all_handle = get_function_handle("shmem_float_min_to_all");
  }

  shmem_float_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __real_shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  typedef void (*shmem_float_prod_to_all_t)(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8);
  static shmem_float_prod_to_all_t shmem_float_prod_to_all_handle = (shmem_float_prod_to_all_t)NULL;
  if (!shmem_float_prod_to_all_handle) {
    shmem_float_prod_to_all_handle = get_function_handle("shmem_float_prod_to_all");
  }

  shmem_float_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {
   __wrap_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_sum_to_all_t)(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8);
  static shmem_double_sum_to_all_t shmem_double_sum_to_all_handle = (shmem_double_sum_to_all_t)NULL;
  if (!shmem_double_sum_to_all_handle) {
    shmem_double_sum_to_all_handle = get_function_handle("shmem_double_sum_to_all");
  }

  shmem_double_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_max_to_all
 **********************************************************/

extern void  __wrap_shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_max_to_all_t)(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8);
  static shmem_double_max_to_all_t shmem_double_max_to_all_handle = (shmem_double_max_to_all_t)NULL;
  if (!shmem_double_max_to_all_handle) {
    shmem_double_max_to_all_handle = get_function_handle("shmem_double_max_to_all");
  }

  shmem_double_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_min_to_all
 **********************************************************/

extern void  __wrap_shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_min_to_all_t)(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8);
  static shmem_double_min_to_all_t shmem_double_min_to_all_handle = (shmem_double_min_to_all_t)NULL;
  if (!shmem_double_min_to_all_handle) {
    shmem_double_min_to_all_handle = get_function_handle("shmem_double_min_to_all");
  }

  shmem_double_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_double_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __real_shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  typedef void (*shmem_double_prod_to_all_t)(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8);
  static shmem_double_prod_to_all_t shmem_double_prod_to_all_handle = (shmem_double_prod_to_all_t)NULL;
  if (!shmem_double_prod_to_all_handle) {
    shmem_double_prod_to_all_handle = get_function_handle("shmem_double_prod_to_all");
  }

  shmem_double_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {
   __wrap_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_ld80_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_ld80_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_ld80_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_ld80_sum_to_all_t)(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8);
  static shmem_ld80_sum_to_all_t shmem_ld80_sum_to_all_handle = (shmem_ld80_sum_to_all_t)NULL;
  if (!shmem_ld80_sum_to_all_handle) {
    shmem_ld80_sum_to_all_handle = get_function_handle("shmem_ld80_sum_to_all");
  }

  shmem_ld80_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_ld80_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_ld80_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_ld80_max_to_all
 **********************************************************/

extern void  __wrap_shmem_ld80_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_ld80_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_ld80_max_to_all_t)(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8);
  static shmem_ld80_max_to_all_t shmem_ld80_max_to_all_handle = (shmem_ld80_max_to_all_t)NULL;
  if (!shmem_ld80_max_to_all_handle) {
    shmem_ld80_max_to_all_handle = get_function_handle("shmem_ld80_max_to_all");
  }

  shmem_ld80_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_ld80_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_ld80_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_ld80_min_to_all
 **********************************************************/

extern void  __wrap_shmem_ld80_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_ld80_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_ld80_min_to_all_t)(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8);
  static shmem_ld80_min_to_all_t shmem_ld80_min_to_all_handle = (shmem_ld80_min_to_all_t)NULL;
  if (!shmem_ld80_min_to_all_handle) {
    shmem_ld80_min_to_all_handle = get_function_handle("shmem_ld80_min_to_all");
  }

  shmem_ld80_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_ld80_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_ld80_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_ld80_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_ld80_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __real_shmem_ld80_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  typedef void (*shmem_ld80_prod_to_all_t)(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8);
  static shmem_ld80_prod_to_all_t shmem_ld80_prod_to_all_handle = (shmem_ld80_prod_to_all_t)NULL;
  if (!shmem_ld80_prod_to_all_handle) {
    shmem_ld80_prod_to_all_handle = get_function_handle("shmem_ld80_prod_to_all");
  }

  shmem_ld80_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_ld80_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {
   __wrap_shmem_ld80_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float128_sum_to_all
 **********************************************************/

extern void  __wrap_shmem_float128_sum_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __real_shmem_float128_sum_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  typedef void (*shmem_float128_sum_to_all_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8);
  static shmem_float128_sum_to_all_t shmem_float128_sum_to_all_handle = (shmem_float128_sum_to_all_t)NULL;
  if (!shmem_float128_sum_to_all_handle) {
    shmem_float128_sum_to_all_handle = get_function_handle("shmem_float128_sum_to_all");
  }

  shmem_float128_sum_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float128_sum_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) {
   __wrap_shmem_float128_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float128_max_to_all
 **********************************************************/

extern void  __wrap_shmem_float128_max_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __real_shmem_float128_max_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  typedef void (*shmem_float128_max_to_all_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8);
  static shmem_float128_max_to_all_t shmem_float128_max_to_all_handle = (shmem_float128_max_to_all_t)NULL;
  if (!shmem_float128_max_to_all_handle) {
    shmem_float128_max_to_all_handle = get_function_handle("shmem_float128_max_to_all");
  }

  shmem_float128_max_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float128_max_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) {
   __wrap_shmem_float128_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float128_min_to_all
 **********************************************************/

extern void  __wrap_shmem_float128_min_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __real_shmem_float128_min_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  typedef void (*shmem_float128_min_to_all_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8);
  static shmem_float128_min_to_all_t shmem_float128_min_to_all_handle = (shmem_float128_min_to_all_t)NULL;
  if (!shmem_float128_min_to_all_handle) {
    shmem_float128_min_to_all_handle = get_function_handle("shmem_float128_min_to_all");
  }

  shmem_float128_min_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float128_min_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) {
   __wrap_shmem_float128_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_float128_prod_to_all
 **********************************************************/

extern void  __wrap_shmem_float128_prod_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __real_shmem_float128_prod_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  typedef void (*shmem_float128_prod_to_all_t)(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8);
  static shmem_float128_prod_to_all_t shmem_float128_prod_to_all_handle = (shmem_float128_prod_to_all_t)NULL;
  if (!shmem_float128_prod_to_all_handle) {
    shmem_float128_prod_to_all_handle = get_function_handle("shmem_float128_prod_to_all");
  }

  shmem_float128_prod_to_all_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_float128_prod_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) {
   __wrap_shmem_float128_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_broadcast32
 **********************************************************/

extern void  __wrap_shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) ;
extern void  __real_shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8)  {

  typedef void (*shmem_broadcast32_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8);
  static shmem_broadcast32_t shmem_broadcast32_handle = (shmem_broadcast32_t)NULL;
  if (!shmem_broadcast32_handle) {
    shmem_broadcast32_handle = get_function_handle("shmem_broadcast32");
  }

  shmem_broadcast32_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) {
   __wrap_shmem_broadcast32(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_broadcast64
 **********************************************************/

extern void  __wrap_shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) ;
extern void  __real_shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8)  {

  typedef void (*shmem_broadcast64_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8);
  static shmem_broadcast64_t shmem_broadcast64_handle = (shmem_broadcast64_t)NULL;
  if (!shmem_broadcast64_handle) {
    shmem_broadcast64_handle = get_function_handle("shmem_broadcast64");
  }

  shmem_broadcast64_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) {
   __wrap_shmem_broadcast64(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_alltoall
 **********************************************************/

extern void  __wrap_shmem_alltoall(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_alltoall(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_alltoall_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  static shmem_alltoall_t shmem_alltoall_handle = (shmem_alltoall_t)NULL;
  if (!shmem_alltoall_handle) {
    shmem_alltoall_handle = get_function_handle("shmem_alltoall");
  }

  shmem_alltoall_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_alltoall(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_alltoall(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_alltoall32
 **********************************************************/

extern void  __wrap_shmem_alltoall32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_alltoall32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_alltoall32_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  static shmem_alltoall32_t shmem_alltoall32_handle = (shmem_alltoall32_t)NULL;
  if (!shmem_alltoall32_handle) {
    shmem_alltoall32_handle = get_function_handle("shmem_alltoall32");
  }

  shmem_alltoall32_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_alltoall32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_alltoall32(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_alltoall64
 **********************************************************/

extern void  __wrap_shmem_alltoall64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_alltoall64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_alltoall64_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  static shmem_alltoall64_t shmem_alltoall64_handle = (shmem_alltoall64_t)NULL;
  if (!shmem_alltoall64_handle) {
    shmem_alltoall64_handle = get_function_handle("shmem_alltoall64");
  }

  shmem_alltoall64_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_alltoall64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_alltoall64(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_team_alltoall
 **********************************************************/

extern void  __wrap_shmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) ;
extern void  __real_shmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)  {

  typedef void (*shmem_team_alltoall_t)(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5);
  static shmem_team_alltoall_t shmem_team_alltoall_handle = (shmem_team_alltoall_t)NULL;
  if (!shmem_team_alltoall_handle) {
    shmem_team_alltoall_handle = get_function_handle("shmem_team_alltoall");
  }

  shmem_team_alltoall_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  shmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) {
   __wrap_shmem_team_alltoall(a1, a2, a3, a4, a5);
}


/**********************************************************
   pshmem_team_alltoall
 **********************************************************/

extern void  __wrap_pshmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) ;
extern void  __real_pshmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)  {

  typedef void (*pshmem_team_alltoall_t)(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5);
  static pshmem_team_alltoall_t pshmem_team_alltoall_handle = (pshmem_team_alltoall_t)NULL;
  if (!pshmem_team_alltoall_handle) {
    pshmem_team_alltoall_handle = get_function_handle("pshmem_team_alltoall");
  }

  pshmem_team_alltoall_handle ( a1,  a2,  a3,  a4,  a5);

}

extern void  pshmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) {
   __wrap_pshmem_team_alltoall(a1, a2, a3, a4, a5);
}


/**********************************************************
   shmem_alltoalls32
 **********************************************************/

extern void  __wrap_shmem_alltoalls32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9) ;
extern void  __real_shmem_alltoalls32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9)  {

  typedef void (*shmem_alltoalls32_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9);
  static shmem_alltoalls32_t shmem_alltoalls32_handle = (shmem_alltoalls32_t)NULL;
  if (!shmem_alltoalls32_handle) {
    shmem_alltoalls32_handle = get_function_handle("shmem_alltoalls32");
  }

  shmem_alltoalls32_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);

}

extern void  shmem_alltoalls32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9) {
   __wrap_shmem_alltoalls32(a1, a2, a3, a4, a5, a6, a7, a8, a9);
}


/**********************************************************
   shmem_alltoalls64
 **********************************************************/

extern void  __wrap_shmem_alltoalls64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9) ;
extern void  __real_shmem_alltoalls64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9)  {

  typedef void (*shmem_alltoalls64_t)(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9);
  static shmem_alltoalls64_t shmem_alltoalls64_handle = (shmem_alltoalls64_t)NULL;
  if (!shmem_alltoalls64_handle) {
    shmem_alltoalls64_handle = get_function_handle("shmem_alltoalls64");
  }

  shmem_alltoalls64_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);

}

extern void  shmem_alltoalls64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9) {
   __wrap_shmem_alltoalls64(a1, a2, a3, a4, a5, a6, a7, a8, a9);
}


/**********************************************************
   shmem_alltoallv
 **********************************************************/

extern void  __wrap_shmem_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) ;
extern void  __real_shmem_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)  {

  typedef void (*shmem_alltoallv_t)(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10);
  static shmem_alltoallv_t shmem_alltoallv_handle = (shmem_alltoallv_t)NULL;
  if (!shmem_alltoallv_handle) {
    shmem_alltoallv_handle = get_function_handle("shmem_alltoallv");
  }

  shmem_alltoallv_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);

}

extern void  shmem_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) {
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}


/**********************************************************
   shmem_team_alltoallv
 **********************************************************/

extern void  __wrap_shmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __real_shmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  typedef void (*shmem_team_alltoallv_t)(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8);
  static shmem_team_alltoallv_t shmem_team_alltoallv_handle = (shmem_team_alltoallv_t)NULL;
  if (!shmem_team_alltoallv_handle) {
    shmem_team_alltoallv_handle = get_function_handle("shmem_team_alltoallv");
  }

  shmem_team_alltoallv_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) {
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   pshmem_team_alltoallv
 **********************************************************/

extern void  __wrap_pshmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __real_pshmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  typedef void (*pshmem_team_alltoallv_t)(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8);
  static pshmem_team_alltoallv_t pshmem_team_alltoallv_handle = (pshmem_team_alltoallv_t)NULL;
  if (!pshmem_team_alltoallv_handle) {
    pshmem_team_alltoallv_handle = get_function_handle("pshmem_team_alltoallv");
  }

  pshmem_team_alltoallv_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  pshmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) {
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_alltoallv_packed
 **********************************************************/

extern void  __wrap_shmem_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) ;
extern void  __real_shmem_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)  {

  typedef void (*shmem_alltoallv_packed_t)(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10);
  static shmem_alltoallv_packed_t shmem_alltoallv_packed_handle = (shmem_alltoallv_packed_t)NULL;
  if (!shmem_alltoallv_packed_handle) {
    shmem_alltoallv_packed_handle = get_function_handle("shmem_alltoallv_packed");
  }

  shmem_alltoallv_packed_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);

}

extern void  shmem_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) {
   __wrap_shmem_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}


/**********************************************************
   shmem_team_alltoallv_packed
 **********************************************************/

extern void  __wrap_shmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __real_shmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  typedef void (*shmem_team_alltoallv_packed_t)(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8);
  static shmem_team_alltoallv_packed_t shmem_team_alltoallv_packed_handle = (shmem_team_alltoallv_packed_t)NULL;
  if (!shmem_team_alltoallv_packed_handle) {
    shmem_team_alltoallv_packed_handle = get_function_handle("shmem_team_alltoallv_packed");
  }

  shmem_team_alltoallv_packed_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  shmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) {
   __wrap_shmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   pshmem_team_alltoallv_packed
 **********************************************************/

extern void  __wrap_pshmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __real_pshmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  typedef void (*pshmem_team_alltoallv_packed_t)(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8);
  static pshmem_team_alltoallv_packed_t pshmem_team_alltoallv_packed_handle = (pshmem_team_alltoallv_packed_t)NULL;
  if (!pshmem_team_alltoallv_packed_handle) {
    pshmem_team_alltoallv_packed_handle = get_function_handle("pshmem_team_alltoallv_packed");
  }

  pshmem_team_alltoallv_packed_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);

}

extern void  pshmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) {
   __wrap_pshmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
}


/**********************************************************
   shmem_collect32
 **********************************************************/

extern void  __wrap_shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_collect32_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  static shmem_collect32_t shmem_collect32_handle = (shmem_collect32_t)NULL;
  if (!shmem_collect32_handle) {
    shmem_collect32_handle = get_function_handle("shmem_collect32");
  }

  shmem_collect32_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_collect32(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_collect64
 **********************************************************/

extern void  __wrap_shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_collect64_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  static shmem_collect64_t shmem_collect64_handle = (shmem_collect64_t)NULL;
  if (!shmem_collect64_handle) {
    shmem_collect64_handle = get_function_handle("shmem_collect64");
  }

  shmem_collect64_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_collect64(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_fcollect32
 **********************************************************/

extern void  __wrap_shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_fcollect32_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  static shmem_fcollect32_t shmem_fcollect32_handle = (shmem_fcollect32_t)NULL;
  if (!shmem_fcollect32_handle) {
    shmem_fcollect32_handle = get_function_handle("shmem_fcollect32");
  }

  shmem_fcollect32_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_fcollect32(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_fcollect64
 **********************************************************/

extern void  __wrap_shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __real_shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  typedef void (*shmem_fcollect64_t)(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7);
  static shmem_fcollect64_t shmem_fcollect64_handle = (shmem_fcollect64_t)NULL;
  if (!shmem_fcollect64_handle) {
    shmem_fcollect64_handle = get_function_handle("shmem_fcollect64");
  }

  shmem_fcollect64_handle ( a1,  a2,  a3,  a4,  a5,  a6,  a7);

}

extern void  shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {
   __wrap_shmem_fcollect64(a1, a2, a3, a4, a5, a6, a7);
}


/**********************************************************
   shmem_team_split
 **********************************************************/

extern void  __wrap_shmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __real_shmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)  {

  typedef void (*shmem_team_split_t)(shmem_team_t a1, int a2, int a3, shmem_team_t * a4);
  static shmem_team_split_t shmem_team_split_handle = (shmem_team_split_t)NULL;
  if (!shmem_team_split_handle) {
    shmem_team_split_handle = get_function_handle("shmem_team_split");
  }

  shmem_team_split_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) {
   __wrap_shmem_team_split(a1, a2, a3, a4);
}


/**********************************************************
   pshmem_team_split
 **********************************************************/

extern void  __wrap_pshmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __real_pshmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)  {

  typedef void (*pshmem_team_split_t)(shmem_team_t a1, int a2, int a3, shmem_team_t * a4);
  static pshmem_team_split_t pshmem_team_split_handle = (pshmem_team_split_t)NULL;
  if (!pshmem_team_split_handle) {
    pshmem_team_split_handle = get_function_handle("pshmem_team_split");
  }

  pshmem_team_split_handle ( a1,  a2,  a3,  a4);

}

extern void  pshmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) {
   __wrap_pshmem_team_split(a1, a2, a3, a4);
}


/**********************************************************
   shmem_team_create_strided
 **********************************************************/

extern void  __wrap_shmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __real_shmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4)  {

  typedef void (*shmem_team_create_strided_t)(int a1, int a2, int a3, shmem_team_t * a4);
  static shmem_team_create_strided_t shmem_team_create_strided_handle = (shmem_team_create_strided_t)NULL;
  if (!shmem_team_create_strided_handle) {
    shmem_team_create_strided_handle = get_function_handle("shmem_team_create_strided");
  }

  shmem_team_create_strided_handle ( a1,  a2,  a3,  a4);

}

extern void  shmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) {
   __wrap_shmem_team_create_strided(a1, a2, a3, a4);
}


/**********************************************************
   pshmem_team_create_strided
 **********************************************************/

extern void  __wrap_pshmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __real_pshmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4)  {

  typedef void (*pshmem_team_create_strided_t)(int a1, int a2, int a3, shmem_team_t * a4);
  static pshmem_team_create_strided_t pshmem_team_create_strided_handle = (pshmem_team_create_strided_t)NULL;
  if (!pshmem_team_create_strided_handle) {
    pshmem_team_create_strided_handle = get_function_handle("pshmem_team_create_strided");
  }

  pshmem_team_create_strided_handle ( a1,  a2,  a3,  a4);

}

extern void  pshmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) {
   __wrap_pshmem_team_create_strided(a1, a2, a3, a4);
}


/**********************************************************
   shmem_team_free
 **********************************************************/

extern void  __wrap_shmem_team_free(shmem_team_t * a1) ;
extern void  __real_shmem_team_free(shmem_team_t * a1)  {

  typedef void (*shmem_team_free_t)(shmem_team_t * a1);
  static shmem_team_free_t shmem_team_free_handle = (shmem_team_free_t)NULL;
  if (!shmem_team_free_handle) {
    shmem_team_free_handle = get_function_handle("shmem_team_free");
  }

  shmem_team_free_handle ( a1);

}

extern void  shmem_team_free(shmem_team_t * a1) {
   __wrap_shmem_team_free(a1);
}


/**********************************************************
   shmem_team_npes
 **********************************************************/

extern int  __wrap_shmem_team_npes(shmem_team_t a1) ;
extern int  __real_shmem_team_npes(shmem_team_t a1)  {

  int retval;
  typedef int (*shmem_team_npes_t)(shmem_team_t a1);
  static shmem_team_npes_t shmem_team_npes_handle = (shmem_team_npes_t)NULL;
  if (!shmem_team_npes_handle) {
    shmem_team_npes_handle = get_function_handle("shmem_team_npes");
  }

  retval  =  shmem_team_npes_handle ( a1);
  return retval;

}

extern int  shmem_team_npes(shmem_team_t a1) {
   __wrap_shmem_team_npes(a1);
}


/**********************************************************
   shmem_team_mype
 **********************************************************/

extern int  __wrap_shmem_team_mype(shmem_team_t a1) ;
extern int  __real_shmem_team_mype(shmem_team_t a1)  {

  int retval;
  typedef int (*shmem_team_mype_t)(shmem_team_t a1);
  static shmem_team_mype_t shmem_team_mype_handle = (shmem_team_mype_t)NULL;
  if (!shmem_team_mype_handle) {
    shmem_team_mype_handle = get_function_handle("shmem_team_mype");
  }

  retval  =  shmem_team_mype_handle ( a1);
  return retval;

}

extern int  shmem_team_mype(shmem_team_t a1) {
   __wrap_shmem_team_mype(a1);
}


/**********************************************************
   shmem_team_translate_pe
 **********************************************************/

extern int  __wrap_shmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) ;
extern int  __real_shmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3)  {

  int retval;
  typedef int (*shmem_team_translate_pe_t)(shmem_team_t a1, int a2, shmem_team_t a3);
  static shmem_team_translate_pe_t shmem_team_translate_pe_handle = (shmem_team_translate_pe_t)NULL;
  if (!shmem_team_translate_pe_handle) {
    shmem_team_translate_pe_handle = get_function_handle("shmem_team_translate_pe");
  }

  retval  =  shmem_team_translate_pe_handle ( a1,  a2,  a3);
  return retval;

}

extern int  shmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) {
   __wrap_shmem_team_translate_pe(a1, a2, a3);
}


/**********************************************************
   pshmem_team_translate_pe
 **********************************************************/

extern int  __wrap_pshmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) ;
extern int  __real_pshmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3)  {

  int retval;
  typedef int (*pshmem_team_translate_pe_t)(shmem_team_t a1, int a2, shmem_team_t a3);
  static pshmem_team_translate_pe_t pshmem_team_translate_pe_handle = (pshmem_team_translate_pe_t)NULL;
  if (!pshmem_team_translate_pe_handle) {
    pshmem_team_translate_pe_handle = get_function_handle("pshmem_team_translate_pe");
  }

  retval  =  pshmem_team_translate_pe_handle ( a1,  a2,  a3);
  return retval;

}

extern int  pshmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) {
   __wrap_pshmem_team_translate_pe(a1, a2, a3);
}


/**********************************************************
   start_pes
 **********************************************************/

extern void  __wrap_start_pes(int a1) ;
extern void  __real_start_pes(int a1)  {

  typedef void (*start_pes_t)(int a1);
  static start_pes_t start_pes_handle = (start_pes_t)NULL;
  if (!start_pes_handle) {
    start_pes_handle = get_function_handle("start_pes");
  }

  start_pes_handle ( a1);

}

extern void  start_pes(int a1) {
   __wrap_start_pes(a1);
}


/**********************************************************
   shmem_init
 **********************************************************/

extern void  __wrap_shmem_init() ;
extern void  __real_shmem_init()  {

  typedef void (*shmem_init_t)();
  static shmem_init_t shmem_init_handle = (shmem_init_t)NULL;
  if (!shmem_init_handle) {
    shmem_init_handle = get_function_handle("shmem_init");
  }

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
  static shmem_finalize_t shmem_finalize_handle = (shmem_finalize_t)NULL;
  if (!shmem_finalize_handle) {
    shmem_finalize_handle = get_function_handle("shmem_finalize");
  }

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
  static shmem_global_exit_t shmem_global_exit_handle = (shmem_global_exit_t)NULL;
  if (!shmem_global_exit_handle) {
    shmem_global_exit_handle = get_function_handle("shmem_global_exit");
  }

  shmem_global_exit_handle ( a1);

}

extern void  shmem_global_exit(int a1) {
   __wrap_shmem_global_exit(a1);
}


/**********************************************************
   _num_pes
 **********************************************************/

extern int  __wrap__num_pes() ;
extern int  __real__num_pes()  {

  int retval;
  typedef int (*_num_pes_t)();
  static _num_pes_t _num_pes_handle = (_num_pes_t)NULL;
  if (!_num_pes_handle) {
    _num_pes_handle = get_function_handle("_num_pes");
  }

  retval  =  _num_pes_handle ();
  return retval;

}

extern int  _num_pes() {
   __wrap__num_pes();
}


/**********************************************************
   shmem_n_pes
 **********************************************************/

extern int  __wrap_shmem_n_pes() ;
extern int  __real_shmem_n_pes()  {

  int retval;
  typedef int (*shmem_n_pes_t)();
  static shmem_n_pes_t shmem_n_pes_handle = (shmem_n_pes_t)NULL;
  if (!shmem_n_pes_handle) {
    shmem_n_pes_handle = get_function_handle("shmem_n_pes");
  }

  retval  =  shmem_n_pes_handle ();
  return retval;

}

extern int  shmem_n_pes() {
   __wrap_shmem_n_pes();
}


/**********************************************************
   _my_pe
 **********************************************************/

extern int  __wrap__my_pe() ;
extern int  __real__my_pe()  {

  int retval;
  typedef int (*_my_pe_t)();
  static _my_pe_t _my_pe_handle = (_my_pe_t)NULL;
  if (!_my_pe_handle) {
    _my_pe_handle = get_function_handle("_my_pe");
  }

  retval  =  _my_pe_handle ();
  return retval;

}

extern int  _my_pe() {
   __wrap__my_pe();
}


/**********************************************************
   shmem_my_pe
 **********************************************************/

extern int  __wrap_shmem_my_pe() ;
extern int  __real_shmem_my_pe()  {

  int retval;
  typedef int (*shmem_my_pe_t)();
  static shmem_my_pe_t shmem_my_pe_handle = (shmem_my_pe_t)NULL;
  if (!shmem_my_pe_handle) {
    shmem_my_pe_handle = get_function_handle("shmem_my_pe");
  }

  retval  =  shmem_my_pe_handle ();
  return retval;

}

extern int  shmem_my_pe() {
   __wrap_shmem_my_pe();
}


/**********************************************************
   shmem_pe_accessible
 **********************************************************/

extern int  __wrap_shmem_pe_accessible(int a1) ;
extern int  __real_shmem_pe_accessible(int a1)  {

  int retval;
  typedef int (*shmem_pe_accessible_t)(int a1);
  static shmem_pe_accessible_t shmem_pe_accessible_handle = (shmem_pe_accessible_t)NULL;
  if (!shmem_pe_accessible_handle) {
    shmem_pe_accessible_handle = get_function_handle("shmem_pe_accessible");
  }

  retval  =  shmem_pe_accessible_handle ( a1);
  return retval;

}

extern int  shmem_pe_accessible(int a1) {
   __wrap_shmem_pe_accessible(a1);
}


/**********************************************************
   shmem_addr_accessible
 **********************************************************/

extern int  __wrap_shmem_addr_accessible(void * a1, int a2) ;
extern int  __real_shmem_addr_accessible(void * a1, int a2)  {

  int retval;
  typedef int (*shmem_addr_accessible_t)(void * a1, int a2);
  static shmem_addr_accessible_t shmem_addr_accessible_handle = (shmem_addr_accessible_t)NULL;
  if (!shmem_addr_accessible_handle) {
    shmem_addr_accessible_handle = get_function_handle("shmem_addr_accessible");
  }

  retval  =  shmem_addr_accessible_handle ( a1,  a2);
  return retval;

}

extern int  shmem_addr_accessible(void * a1, int a2) {
   __wrap_shmem_addr_accessible(a1, a2);
}


/**********************************************************
   shmem_init_thread
 **********************************************************/

extern int  __wrap_shmem_init_thread(int a1) ;
extern int  __real_shmem_init_thread(int a1)  {

  int retval;
  typedef int (*shmem_init_thread_t)(int a1);
  static shmem_init_thread_t shmem_init_thread_handle = (shmem_init_thread_t)NULL;
  if (!shmem_init_thread_handle) {
    shmem_init_thread_handle = get_function_handle("shmem_init_thread");
  }

  retval  =  shmem_init_thread_handle ( a1);
  return retval;

}

extern int  shmem_init_thread(int a1) {
   __wrap_shmem_init_thread(a1);
}


/**********************************************************
   shmem_query_thread
 **********************************************************/

extern int  __wrap_shmem_query_thread() ;
extern int  __real_shmem_query_thread()  {

  int retval;
  typedef int (*shmem_query_thread_t)();
  static shmem_query_thread_t shmem_query_thread_handle = (shmem_query_thread_t)NULL;
  if (!shmem_query_thread_handle) {
    shmem_query_thread_handle = get_function_handle("shmem_query_thread");
  }

  retval  =  shmem_query_thread_handle ();
  return retval;

}

extern int  shmem_query_thread() {
   __wrap_shmem_query_thread();
}


/**********************************************************
   shmem_thread_register
 **********************************************************/

extern void  __wrap_shmem_thread_register() ;
extern void  __real_shmem_thread_register()  {

  typedef void (*shmem_thread_register_t)();
  static shmem_thread_register_t shmem_thread_register_handle = (shmem_thread_register_t)NULL;
  if (!shmem_thread_register_handle) {
    shmem_thread_register_handle = get_function_handle("shmem_thread_register");
  }

  shmem_thread_register_handle ();

}

extern void  shmem_thread_register() {
   __wrap_shmem_thread_register();
}


/**********************************************************
   shmem_thread_unregister
 **********************************************************/

extern void  __wrap_shmem_thread_unregister() ;
extern void  __real_shmem_thread_unregister()  {

  typedef void (*shmem_thread_unregister_t)();
  static shmem_thread_unregister_t shmem_thread_unregister_handle = (shmem_thread_unregister_t)NULL;
  if (!shmem_thread_unregister_handle) {
    shmem_thread_unregister_handle = get_function_handle("shmem_thread_unregister");
  }

  shmem_thread_unregister_handle ();

}

extern void  shmem_thread_unregister() {
   __wrap_shmem_thread_unregister();
}


/**********************************************************
   shmem_thread_fence
 **********************************************************/

extern void  __wrap_shmem_thread_fence() ;
extern void  __real_shmem_thread_fence()  {

  typedef void (*shmem_thread_fence_t)();
  static shmem_thread_fence_t shmem_thread_fence_handle = (shmem_thread_fence_t)NULL;
  if (!shmem_thread_fence_handle) {
    shmem_thread_fence_handle = get_function_handle("shmem_thread_fence");
  }

  shmem_thread_fence_handle ();

}

extern void  shmem_thread_fence() {
   __wrap_shmem_thread_fence();
}


/**********************************************************
   shmem_thread_quiet
 **********************************************************/

extern void  __wrap_shmem_thread_quiet() ;
extern void  __real_shmem_thread_quiet()  {

  typedef void (*shmem_thread_quiet_t)();
  static shmem_thread_quiet_t shmem_thread_quiet_handle = (shmem_thread_quiet_t)NULL;
  if (!shmem_thread_quiet_handle) {
    shmem_thread_quiet_handle = get_function_handle("shmem_thread_quiet");
  }

  shmem_thread_quiet_handle ();

}

extern void  shmem_thread_quiet() {
   __wrap_shmem_thread_quiet();
}


/**********************************************************
   shmem_local_npes
 **********************************************************/

extern int  __wrap_shmem_local_npes() ;
extern int  __real_shmem_local_npes()  {

  int retval;
  typedef int (*shmem_local_npes_t)();
  static shmem_local_npes_t shmem_local_npes_handle = (shmem_local_npes_t)NULL;
  if (!shmem_local_npes_handle) {
    shmem_local_npes_handle = get_function_handle("shmem_local_npes");
  }

  retval  =  shmem_local_npes_handle ();
  return retval;

}

extern int  shmem_local_npes() {
   __wrap_shmem_local_npes();
}


/**********************************************************
   shmem_local_pes
 **********************************************************/

extern void  __wrap_shmem_local_pes(int * a1, int a2) ;
extern void  __real_shmem_local_pes(int * a1, int a2)  {

  typedef void (*shmem_local_pes_t)(int * a1, int a2);
  static shmem_local_pes_t shmem_local_pes_handle = (shmem_local_pes_t)NULL;
  if (!shmem_local_pes_handle) {
    shmem_local_pes_handle = get_function_handle("shmem_local_pes");
  }

  shmem_local_pes_handle ( a1,  a2);

}

extern void  shmem_local_pes(int * a1, int a2) {
   __wrap_shmem_local_pes(a1, a2);
}


/**********************************************************
   shmem_set_cache_inv
 **********************************************************/

extern void  __wrap_shmem_set_cache_inv() ;
extern void  __real_shmem_set_cache_inv()  {

  typedef void (*shmem_set_cache_inv_t)();
  static shmem_set_cache_inv_t shmem_set_cache_inv_handle = (shmem_set_cache_inv_t)NULL;
  if (!shmem_set_cache_inv_handle) {
    shmem_set_cache_inv_handle = get_function_handle("shmem_set_cache_inv");
  }

  shmem_set_cache_inv_handle ();

}

extern void  shmem_set_cache_inv() {
   __wrap_shmem_set_cache_inv();
}


/**********************************************************
   shmem_set_cache_line_inv
 **********************************************************/

extern void  __wrap_shmem_set_cache_line_inv(void * a1) ;
extern void  __real_shmem_set_cache_line_inv(void * a1)  {

  typedef void (*shmem_set_cache_line_inv_t)(void * a1);
  static shmem_set_cache_line_inv_t shmem_set_cache_line_inv_handle = (shmem_set_cache_line_inv_t)NULL;
  if (!shmem_set_cache_line_inv_handle) {
    shmem_set_cache_line_inv_handle = get_function_handle("shmem_set_cache_line_inv");
  }

  shmem_set_cache_line_inv_handle ( a1);

}

extern void  shmem_set_cache_line_inv(void * a1) {
   __wrap_shmem_set_cache_line_inv(a1);
}


/**********************************************************
   shmem_clear_cache_inv
 **********************************************************/

extern void  __wrap_shmem_clear_cache_inv() ;
extern void  __real_shmem_clear_cache_inv()  {

  typedef void (*shmem_clear_cache_inv_t)();
  static shmem_clear_cache_inv_t shmem_clear_cache_inv_handle = (shmem_clear_cache_inv_t)NULL;
  if (!shmem_clear_cache_inv_handle) {
    shmem_clear_cache_inv_handle = get_function_handle("shmem_clear_cache_inv");
  }

  shmem_clear_cache_inv_handle ();

}

extern void  shmem_clear_cache_inv() {
   __wrap_shmem_clear_cache_inv();
}


/**********************************************************
   shmem_clear_cache_line_inv
 **********************************************************/

extern void  __wrap_shmem_clear_cache_line_inv(void * a1) ;
extern void  __real_shmem_clear_cache_line_inv(void * a1)  {

  typedef void (*shmem_clear_cache_line_inv_t)(void * a1);
  static shmem_clear_cache_line_inv_t shmem_clear_cache_line_inv_handle = (shmem_clear_cache_line_inv_t)NULL;
  if (!shmem_clear_cache_line_inv_handle) {
    shmem_clear_cache_line_inv_handle = get_function_handle("shmem_clear_cache_line_inv");
  }

  shmem_clear_cache_line_inv_handle ( a1);

}

extern void  shmem_clear_cache_line_inv(void * a1) {
   __wrap_shmem_clear_cache_line_inv(a1);
}


/**********************************************************
   shmem_udcflush
 **********************************************************/

extern void  __wrap_shmem_udcflush() ;
extern void  __real_shmem_udcflush()  {

  typedef void (*shmem_udcflush_t)();
  static shmem_udcflush_t shmem_udcflush_handle = (shmem_udcflush_t)NULL;
  if (!shmem_udcflush_handle) {
    shmem_udcflush_handle = get_function_handle("shmem_udcflush");
  }

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
  static shmem_udcflush_line_t shmem_udcflush_line_handle = (shmem_udcflush_line_t)NULL;
  if (!shmem_udcflush_line_handle) {
    shmem_udcflush_line_handle = get_function_handle("shmem_udcflush_line");
  }

  shmem_udcflush_line_handle ( a1);

}

extern void  shmem_udcflush_line(void * a1) {
   __wrap_shmem_udcflush_line(a1);
}


/**********************************************************
   shfree
 **********************************************************/

extern void  __wrap_shfree(void * a1) ;
extern void  __real_shfree(void * a1)  {

  typedef void (*shfree_t)(void * a1);
  static shfree_t shfree_handle = (shfree_t)NULL;
  if (!shfree_handle) {
    shfree_handle = get_function_handle("shfree");
  }

  shfree_handle ( a1);

}

extern void  shfree(void * a1) {
   __wrap_shfree(a1);
}


/**********************************************************
   shmem_free
 **********************************************************/

extern void  __wrap_shmem_free(void * a1) ;
extern void  __real_shmem_free(void * a1)  {

  typedef void (*shmem_free_t)(void * a1);
  static shmem_free_t shmem_free_handle = (shmem_free_t)NULL;
  if (!shmem_free_handle) {
    shmem_free_handle = get_function_handle("shmem_free");
  }

  shmem_free_handle ( a1);

}

extern void  shmem_free(void * a1) {
   __wrap_shmem_free(a1);
}

