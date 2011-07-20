#include <TAU.h>
#include <shmem.h>
int TAUDECL tau_totalnodes(int set_or_get, int value);
int tau_shmem_tagid=0 ; 
#define TAU_SHMEM_TAGID tau_shmem_tagid
#define TAU_SHMEM_TAGID_NEXT (++tau_shmem_tagid) % 256

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get16 wrapper function 
******************************************************/
void shmem_get16( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get16( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get16( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get16 wrapper function 
******************************************************/
void SHMEM_GET16( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get16( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get16( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get16 wrapper function 
******************************************************/
void shmem_get16_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get16( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get16( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get16 wrapper function 
******************************************************/
void shmem_get16__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get16( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get16( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get32 wrapper function 
******************************************************/
void shmem_get32( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32 wrapper function 
******************************************************/
void SHMEM_GET32( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32 wrapper function 
******************************************************/
void shmem_get32_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32 wrapper function 
******************************************************/
void shmem_get32__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get64 wrapper function 
******************************************************/
void shmem_get64( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64 wrapper function 
******************************************************/
void SHMEM_GET64( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64 wrapper function 
******************************************************/
void shmem_get64_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64 wrapper function 
******************************************************/
void shmem_get64__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get128 wrapper function 
******************************************************/
void shmem_get128( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128 wrapper function 
******************************************************/
void SHMEM_GET128( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128 wrapper function 
******************************************************/
void shmem_get128_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128 wrapper function 
******************************************************/
void shmem_get128__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_getmem wrapper function 
******************************************************/
void shmem_getmem( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem wrapper function 
******************************************************/
void SHMEM_GETMEM( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem wrapper function 
******************************************************/
void shmem_getmem_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem wrapper function 
******************************************************/
void shmem_getmem__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_get wrapper function 
******************************************************/
void shmem_short_get( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get wrapper function 
******************************************************/
void SHMEM_SHORT_GET( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get wrapper function 
******************************************************/
void shmem_short_get_( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get wrapper function 
******************************************************/
void shmem_short_get__( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_get wrapper function 
******************************************************/
void shmem_int_get( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get wrapper function 
******************************************************/
void SHMEM_INT_GET( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get wrapper function 
******************************************************/
void shmem_int_get_( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get wrapper function 
******************************************************/
void shmem_int_get__( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_get wrapper function 
******************************************************/
void shmem_long_get( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get wrapper function 
******************************************************/
void SHMEM_LONG_GET( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get wrapper function 
******************************************************/
void shmem_long_get_( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get wrapper function 
******************************************************/
void shmem_long_get__( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_get wrapper function 
******************************************************/
void shmem_longlong_get( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get wrapper function 
******************************************************/
void SHMEM_LONGLONG_GET( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get wrapper function 
******************************************************/
void shmem_longlong_get_( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get wrapper function 
******************************************************/
void shmem_longlong_get__( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_get wrapper function 
******************************************************/
void shmem_float_get( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get wrapper function 
******************************************************/
void SHMEM_FLOAT_GET( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get wrapper function 
******************************************************/
void shmem_float_get_( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get wrapper function 
******************************************************/
void shmem_float_get__( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_get wrapper function 
******************************************************/
void shmem_double_get( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get wrapper function 
******************************************************/
void SHMEM_DOUBLE_GET( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get wrapper function 
******************************************************/
void shmem_double_get_( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get wrapper function 
******************************************************/
void shmem_double_get__( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put16 wrapper function 
******************************************************/
void shmem_put16( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16 wrapper function 
******************************************************/
void SHMEM_PUT16( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16 wrapper function 
******************************************************/
void shmem_put16_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16 wrapper function 
******************************************************/
void shmem_put16__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put32 wrapper function 
******************************************************/
void shmem_put32( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32 wrapper function 
******************************************************/
void SHMEM_PUT32( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32 wrapper function 
******************************************************/
void shmem_put32_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32 wrapper function 
******************************************************/
void shmem_put32__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put64 wrapper function 
******************************************************/
void shmem_put64( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64 wrapper function 
******************************************************/
void SHMEM_PUT64( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64 wrapper function 
******************************************************/
void shmem_put64_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64 wrapper function 
******************************************************/
void shmem_put64__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put128 wrapper function 
******************************************************/
void shmem_put128( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128 wrapper function 
******************************************************/
void SHMEM_PUT128( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128 wrapper function 
******************************************************/
void shmem_put128_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128 wrapper function 
******************************************************/
void shmem_put128__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_putmem wrapper function 
******************************************************/
void shmem_putmem( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem wrapper function 
******************************************************/
void SHMEM_PUTMEM( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem wrapper function 
******************************************************/
void shmem_putmem_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem wrapper function 
******************************************************/
void shmem_putmem__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_put wrapper function 
******************************************************/
void shmem_short_put( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put wrapper function 
******************************************************/
void SHMEM_SHORT_PUT( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put wrapper function 
******************************************************/
void shmem_short_put_( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put wrapper function 
******************************************************/
void shmem_short_put__( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_put wrapper function 
******************************************************/
void shmem_int_put( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put wrapper function 
******************************************************/
void SHMEM_INT_PUT( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put wrapper function 
******************************************************/
void shmem_int_put_( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put wrapper function 
******************************************************/
void shmem_int_put__( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/

/******************************************************
***      shmem_long_put wrapper function 
******************************************************/
void shmem_long_put( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, pe, sizeof(long)*len); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID,Tau_get_node(), sizeof(long)*len, pe);
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put wrapper function 
******************************************************/
void SHMEM_LONG_PUT( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put wrapper function 
******************************************************/
void shmem_long_put_( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put wrapper function 
******************************************************/
void shmem_long_put__( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_put wrapper function 
******************************************************/
void shmem_longlong_put( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put wrapper function 
******************************************************/
void SHMEM_LONGLONG_PUT( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put wrapper function 
******************************************************/
void shmem_longlong_put_( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put wrapper function 
******************************************************/
void shmem_longlong_put__( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_put wrapper function 
******************************************************/
void shmem_float_put( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put wrapper function 
******************************************************/
void SHMEM_FLOAT_PUT( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put wrapper function 
******************************************************/
void shmem_float_put_( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put wrapper function 
******************************************************/
void shmem_float_put__( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_put wrapper function 
******************************************************/
void shmem_double_put( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put wrapper function 
******************************************************/
void SHMEM_DOUBLE_PUT( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put wrapper function 
******************************************************/
void shmem_double_put_( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put wrapper function 
******************************************************/
void shmem_double_put__( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put16_nb wrapper function 
******************************************************/
void shmem_put16_nb( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put16_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16_nb wrapper function 
******************************************************/
void SHMEM_PUT16_NB( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put16_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16_nb wrapper function 
******************************************************/
void shmem_put16_nb_( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put16_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16_nb wrapper function 
******************************************************/
void shmem_put16_nb__( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put16_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put32_nb wrapper function 
******************************************************/
void shmem_put32_nb( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put32_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32_nb wrapper function 
******************************************************/
void SHMEM_PUT32_NB( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put32_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32_nb wrapper function 
******************************************************/
void shmem_put32_nb_( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put32_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32_nb wrapper function 
******************************************************/
void shmem_put32_nb__( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put32_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put64_nb wrapper function 
******************************************************/
void shmem_put64_nb( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put64_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64_nb wrapper function 
******************************************************/
void SHMEM_PUT64_NB( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put64_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64_nb wrapper function 
******************************************************/
void shmem_put64_nb_( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put64_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64_nb wrapper function 
******************************************************/
void shmem_put64_nb__( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put64_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put128_nb wrapper function 
******************************************************/
void shmem_put128_nb( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put128_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128_nb wrapper function 
******************************************************/
void SHMEM_PUT128_NB( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put128_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128_nb wrapper function 
******************************************************/
void shmem_put128_nb_( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put128_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128_nb wrapper function 
******************************************************/
void shmem_put128_nb__( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put128_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_putmem_nb wrapper function 
******************************************************/
void shmem_putmem_nb( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem_nb wrapper function 
******************************************************/
void SHMEM_PUTMEM_NB( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem_nb wrapper function 
******************************************************/
void shmem_putmem_nb_( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem_nb wrapper function 
******************************************************/
void shmem_putmem_nb__( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_put_nb wrapper function 
******************************************************/
void shmem_short_put_nb( short * trg, const short * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put_nb wrapper function 
******************************************************/
void SHMEM_SHORT_PUT_NB( short * trg, const short * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put_nb wrapper function 
******************************************************/
void shmem_short_put_nb_( short * trg, const short * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put_nb wrapper function 
******************************************************/
void shmem_short_put_nb__( short * trg, const short * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_put_nb wrapper function 
******************************************************/
void shmem_int_put_nb( int * trg, const int * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put_nb wrapper function 
******************************************************/
void SHMEM_INT_PUT_NB( int * trg, const int * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put_nb wrapper function 
******************************************************/
void shmem_int_put_nb_( int * trg, const int * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put_nb wrapper function 
******************************************************/
void shmem_int_put_nb__( int * trg, const int * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_put_nb wrapper function 
******************************************************/
void shmem_long_put_nb( long * trg, const long * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put_nb wrapper function 
******************************************************/
void SHMEM_LONG_PUT_NB( long * trg, const long * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put_nb wrapper function 
******************************************************/
void shmem_long_put_nb_( long * trg, const long * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put_nb wrapper function 
******************************************************/
void shmem_long_put_nb__( long * trg, const long * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_put_nb wrapper function 
******************************************************/
void shmem_longlong_put_nb( long long * trg, const long long * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put_nb wrapper function 
******************************************************/
void SHMEM_LONGLONG_PUT_NB( long long * trg, const long long * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put_nb wrapper function 
******************************************************/
void shmem_longlong_put_nb_( long long * trg, const long long * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put_nb wrapper function 
******************************************************/
void shmem_longlong_put_nb__( long long * trg, const long long * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_put_nb wrapper function 
******************************************************/
void shmem_float_put_nb( float * trg, const float * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put_nb wrapper function 
******************************************************/
void SHMEM_FLOAT_PUT_NB( float * trg, const float * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put_nb wrapper function 
******************************************************/
void shmem_float_put_nb_( float * trg, const float * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put_nb wrapper function 
******************************************************/
void shmem_float_put_nb__( float * trg, const float * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_put_nb wrapper function 
******************************************************/
void shmem_double_put_nb( double * trg, const double * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put_nb wrapper function 
******************************************************/
void SHMEM_DOUBLE_PUT_NB( double * trg, const double * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put_nb wrapper function 
******************************************************/
void shmem_double_put_nb_( double * trg, const double * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put_nb wrapper function 
******************************************************/
void shmem_double_put_nb__( double * trg, const double * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_iget wrapper function 
******************************************************/
void shmem_short_iget( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget wrapper function 
******************************************************/
void SHMEM_SHORT_IGET( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget wrapper function 
******************************************************/
void shmem_short_iget_( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget wrapper function 
******************************************************/
void shmem_short_iget__( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_iget wrapper function 
******************************************************/
void shmem_int_iget( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget wrapper function 
******************************************************/
void SHMEM_INT_IGET( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget wrapper function 
******************************************************/
void shmem_int_iget_( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget wrapper function 
******************************************************/
void shmem_int_iget__( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_iget wrapper function 
******************************************************/
void shmem_long_iget( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget wrapper function 
******************************************************/
void SHMEM_LONG_IGET( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget wrapper function 
******************************************************/
void shmem_long_iget_( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget wrapper function 
******************************************************/
void shmem_long_iget__( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_iget wrapper function 
******************************************************/
void shmem_longlong_iget( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget wrapper function 
******************************************************/
void SHMEM_LONGLONG_IGET( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget wrapper function 
******************************************************/
void shmem_longlong_iget_( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget wrapper function 
******************************************************/
void shmem_longlong_iget__( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_iget wrapper function 
******************************************************/
void shmem_float_iget( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget wrapper function 
******************************************************/
void SHMEM_FLOAT_IGET( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget wrapper function 
******************************************************/
void shmem_float_iget_( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget wrapper function 
******************************************************/
void shmem_float_iget__( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_iget wrapper function 
******************************************************/
void shmem_double_iget( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget wrapper function 
******************************************************/
void SHMEM_DOUBLE_IGET( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget wrapper function 
******************************************************/
void shmem_double_iget_( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget wrapper function 
******************************************************/
void shmem_double_iget__( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iget16 wrapper function 
******************************************************/
void shmem_iget16( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget16( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget16( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget16 wrapper function 
******************************************************/
void SHMEM_IGET16( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget16( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget16( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget16 wrapper function 
******************************************************/
void shmem_iget16_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget16( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget16( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget16 wrapper function 
******************************************************/
void shmem_iget16__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget16( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget16( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iget32 wrapper function 
******************************************************/
void shmem_iget32( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32 wrapper function 
******************************************************/
void SHMEM_IGET32( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32 wrapper function 
******************************************************/
void shmem_iget32_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32 wrapper function 
******************************************************/
void shmem_iget32__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iget64 wrapper function 
******************************************************/
void shmem_iget64( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64 wrapper function 
******************************************************/
void SHMEM_IGET64( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64 wrapper function 
******************************************************/
void shmem_iget64_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64 wrapper function 
******************************************************/
void shmem_iget64__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iget128 wrapper function 
******************************************************/
void shmem_iget128( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128 wrapper function 
******************************************************/
void SHMEM_IGET128( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128 wrapper function 
******************************************************/
void shmem_iget128_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128 wrapper function 
******************************************************/
void shmem_iget128__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_iput wrapper function 
******************************************************/
void shmem_short_iput( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput wrapper function 
******************************************************/
void SHMEM_SHORT_IPUT( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput wrapper function 
******************************************************/
void shmem_short_iput_( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput wrapper function 
******************************************************/
void shmem_short_iput__( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_iput wrapper function 
******************************************************/
void shmem_int_iput( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput wrapper function 
******************************************************/
void SHMEM_INT_IPUT( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput wrapper function 
******************************************************/
void shmem_int_iput_( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput wrapper function 
******************************************************/
void shmem_int_iput__( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_iput wrapper function 
******************************************************/
void shmem_long_iput( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput wrapper function 
******************************************************/
void SHMEM_LONG_IPUT( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput wrapper function 
******************************************************/
void shmem_long_iput_( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput wrapper function 
******************************************************/
void shmem_long_iput__( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_iput wrapper function 
******************************************************/
void shmem_longlong_iput( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput wrapper function 
******************************************************/
void SHMEM_LONGLONG_IPUT( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput wrapper function 
******************************************************/
void shmem_longlong_iput_( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput wrapper function 
******************************************************/
void shmem_longlong_iput__( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_iput wrapper function 
******************************************************/
void shmem_float_iput( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput wrapper function 
******************************************************/
void SHMEM_FLOAT_IPUT( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput wrapper function 
******************************************************/
void shmem_float_iput_( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput wrapper function 
******************************************************/
void shmem_float_iput__( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_iput wrapper function 
******************************************************/
void shmem_double_iput( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput wrapper function 
******************************************************/
void SHMEM_DOUBLE_IPUT( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput wrapper function 
******************************************************/
void shmem_double_iput_( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput wrapper function 
******************************************************/
void shmem_double_iput__( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput16 wrapper function 
******************************************************/
void shmem_iput16( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput16( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput16( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput16 wrapper function 
******************************************************/
void SHMEM_IPUT16( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput16( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput16( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput16 wrapper function 
******************************************************/
void shmem_iput16_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput16( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput16( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput16 wrapper function 
******************************************************/
void shmem_iput16__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput16()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput16( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput16( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput32 wrapper function 
******************************************************/
void shmem_iput32( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32 wrapper function 
******************************************************/
void SHMEM_IPUT32( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32 wrapper function 
******************************************************/
void shmem_iput32_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32 wrapper function 
******************************************************/
void shmem_iput32__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput64 wrapper function 
******************************************************/
void shmem_iput64( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64 wrapper function 
******************************************************/
void SHMEM_IPUT64( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64 wrapper function 
******************************************************/
void shmem_iput64_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64 wrapper function 
******************************************************/
void shmem_iput64__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput128 wrapper function 
******************************************************/
void shmem_iput128( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128 wrapper function 
******************************************************/
void SHMEM_IPUT128( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128 wrapper function 
******************************************************/
void shmem_iput128_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128 wrapper function 
******************************************************/
void shmem_iput128__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_g wrapper function 
******************************************************/
char shmem_char_g( const char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_g wrapper function 
******************************************************/
char SHMEM_CHAR_G( const char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_g wrapper function 
******************************************************/
char shmem_char_g_( const char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_g wrapper function 
******************************************************/
char shmem_char_g__( const char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_g wrapper function 
******************************************************/
short shmem_short_g( const short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_g wrapper function 
******************************************************/
short SHMEM_SHORT_G( const short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_g wrapper function 
******************************************************/
short shmem_short_g_( const short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_g wrapper function 
******************************************************/
short shmem_short_g__( const short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_g wrapper function 
******************************************************/
int shmem_int_g( const int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_g wrapper function 
******************************************************/
int SHMEM_INT_G( const int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_g wrapper function 
******************************************************/
int shmem_int_g_( const int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_g wrapper function 
******************************************************/
int shmem_int_g__( const int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_g wrapper function 
******************************************************/
long shmem_long_g( const long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_g wrapper function 
******************************************************/
long SHMEM_LONG_G( const long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_g wrapper function 
******************************************************/
long shmem_long_g_( const long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_g wrapper function 
******************************************************/
long shmem_long_g__( const long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_g wrapper function 
******************************************************/
long long shmem_longlong_g( const long long * addr, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_g wrapper function 
******************************************************/
long long SHMEM_LONGLONG_G( const long long * addr, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_g wrapper function 
******************************************************/
long long shmem_longlong_g_( const long long * addr, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_g wrapper function 
******************************************************/
long long shmem_longlong_g__( const long long * addr, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_g wrapper function 
******************************************************/
float shmem_float_g( const float * addr, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_g wrapper function 
******************************************************/
float SHMEM_FLOAT_G( const float * addr, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_g wrapper function 
******************************************************/
float shmem_float_g_( const float * addr, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_g wrapper function 
******************************************************/
float shmem_float_g__( const float * addr, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_g wrapper function 
******************************************************/
double shmem_double_g( const double * addr, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_g wrapper function 
******************************************************/
double SHMEM_DOUBLE_G( const double * addr, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_g wrapper function 
******************************************************/
double shmem_double_g_( const double * addr, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_g wrapper function 
******************************************************/
double shmem_double_g__( const double * addr, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_p wrapper function 
******************************************************/
void shmem_char_p( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_p wrapper function 
******************************************************/
void SHMEM_CHAR_P( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_p wrapper function 
******************************************************/
void shmem_char_p_( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_p wrapper function 
******************************************************/
void shmem_char_p__( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_p wrapper function 
******************************************************/
void shmem_short_p( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_p wrapper function 
******************************************************/
void SHMEM_SHORT_P( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_p wrapper function 
******************************************************/
void shmem_short_p_( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_p wrapper function 
******************************************************/
void shmem_short_p__( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_p wrapper function 
******************************************************/
void shmem_int_p( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_p wrapper function 
******************************************************/
void SHMEM_INT_P( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_p wrapper function 
******************************************************/
void shmem_int_p_( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_p wrapper function 
******************************************************/
void shmem_int_p__( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_p wrapper function 
******************************************************/
void shmem_long_p( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_p wrapper function 
******************************************************/
void SHMEM_LONG_P( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_p wrapper function 
******************************************************/
void shmem_long_p_( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_p wrapper function 
******************************************************/
void shmem_long_p__( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_p wrapper function 
******************************************************/
void shmem_longlong_p( long long * addr, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_p wrapper function 
******************************************************/
void SHMEM_LONGLONG_P( long long * addr, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_p wrapper function 
******************************************************/
void shmem_longlong_p_( long long * addr, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_p wrapper function 
******************************************************/
void shmem_longlong_p__( long long * addr, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_p wrapper function 
******************************************************/
void shmem_float_p( float * addr, float value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_p wrapper function 
******************************************************/
void SHMEM_FLOAT_P( float * addr, float value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_p wrapper function 
******************************************************/
void shmem_float_p_( float * addr, float value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_p wrapper function 
******************************************************/
void shmem_float_p__( float * addr, float value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_p wrapper function 
******************************************************/
void shmem_double_p( double * addr, double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_p wrapper function 
******************************************************/
void SHMEM_DOUBLE_P( double * addr, double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_p wrapper function 
******************************************************/
void shmem_double_p_( double * addr, double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_p wrapper function 
******************************************************/
void shmem_double_p__( double * addr, double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_swap wrapper function 
******************************************************/
int shmem_int_swap( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap wrapper function 
******************************************************/
int SHMEM_INT_SWAP( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap wrapper function 
******************************************************/
int shmem_int_swap_( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap wrapper function 
******************************************************/
int shmem_int_swap__( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_swap wrapper function 
******************************************************/
long shmem_long_swap( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap wrapper function 
******************************************************/
long SHMEM_LONG_SWAP( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap wrapper function 
******************************************************/
long shmem_long_swap_( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap wrapper function 
******************************************************/
long shmem_long_swap__( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


#ifndef TAU_CATAMOUNT
#ifndef TAU_CRAYCNL
/******************************************************
***      shmem_float_swap wrapper function 
******************************************************/
float shmem_float_swap( float * addr, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap wrapper function 
******************************************************/
float SHMEM_FLOAT_SWAP( float * addr, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap wrapper function 
******************************************************/
float shmem_float_swap_( float * addr, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap wrapper function 
******************************************************/
float shmem_float_swap__( float * addr, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_swap wrapper function 
******************************************************/
double shmem_double_swap( double * addr, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap wrapper function 
******************************************************/
double SHMEM_DOUBLE_SWAP( double * addr, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap wrapper function 
******************************************************/
double shmem_double_swap_( double * addr, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap wrapper function 
******************************************************/
double shmem_double_swap__( double * addr, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_cswap wrapper function 
******************************************************/
int shmem_int_cswap( int * addr, int match, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap wrapper function 
******************************************************/
int SHMEM_INT_CSWAP( int * addr, int match, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap wrapper function 
******************************************************/
int shmem_int_cswap_( int * addr, int match, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap wrapper function 
******************************************************/
int shmem_int_cswap__( int * addr, int match, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_cswap wrapper function 
******************************************************/
long shmem_long_cswap( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap wrapper function 
******************************************************/
long SHMEM_LONG_CSWAP( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap wrapper function 
******************************************************/
long shmem_long_cswap_( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap wrapper function 
******************************************************/
long shmem_long_cswap__( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/

#endif /* TAU_CRAYCNL */
#endif /* TAU_CATAMOUNT */

/******************************************************
***      shmem_int_finc wrapper function 
******************************************************/
int shmem_int_finc( int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc wrapper function 
******************************************************/
int SHMEM_INT_FINC( int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc wrapper function 
******************************************************/
int shmem_int_finc_( int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc wrapper function 
******************************************************/
int shmem_int_finc__( int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_finc wrapper function 
******************************************************/
long shmem_long_finc( long * target, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_finc wrapper function 
******************************************************/
long SHMEM_LONG_FINC( long * target, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_finc wrapper function 
******************************************************/
long shmem_long_finc_( long * target, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_finc wrapper function 
******************************************************/
long shmem_long_finc__( long * target, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_fadd wrapper function 
******************************************************/
int shmem_int_fadd( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd wrapper function 
******************************************************/
int SHMEM_INT_FADD( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd wrapper function 
******************************************************/
int shmem_int_fadd_( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd wrapper function 
******************************************************/
int shmem_int_fadd__( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_fadd wrapper function 
******************************************************/
long shmem_long_fadd( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_fadd wrapper function 
******************************************************/
long SHMEM_LONG_FADD( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_fadd wrapper function 
******************************************************/
long shmem_long_fadd_( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_fadd wrapper function 
******************************************************/
long shmem_long_fadd__( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_barrier_all wrapper function 
******************************************************/
void shmem_barrier_all( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_all( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_all( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_all wrapper function 
******************************************************/
void SHMEM_BARRIER_ALL( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_all( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_all( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_all wrapper function 
******************************************************/
void shmem_barrier_all_( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_all( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_all( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_all wrapper function 
******************************************************/
void shmem_barrier_all__( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_all( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_all( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_barrier wrapper function 
******************************************************/
void shmem_barrier( int pestart, int log_pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier( pestart, log_pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier( pestart, log_pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier wrapper function 
******************************************************/
void SHMEM_BARRIER( int pestart, int log_pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier( pestart, log_pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier( pestart, log_pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier wrapper function 
******************************************************/
void shmem_barrier_( int pestart, int log_pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier( pestart, log_pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier( pestart, log_pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier wrapper function 
******************************************************/
void shmem_barrier__( int pestart, int log_pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier( pestart, log_pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier( pestart, log_pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_quiet wrapper function 
******************************************************/
void shmem_quiet( )
{

  TAU_PROFILE_TIMER(t, "shmem_quiet()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_quiet( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_quiet( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_quiet wrapper function 
******************************************************/
void SHMEM_QUIET( )
{

  TAU_PROFILE_TIMER(t, "shmem_quiet()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_quiet( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_quiet( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_quiet wrapper function 
******************************************************/
void shmem_quiet_( )
{

  TAU_PROFILE_TIMER(t, "shmem_quiet()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_quiet( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_quiet( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_quiet wrapper function 
******************************************************/
void shmem_quiet__( )
{

  TAU_PROFILE_TIMER(t, "shmem_quiet()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_quiet( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_quiet( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/

#ifndef TAU_CATAMOUNT 

/******************************************************
***      shmem_set_lock wrapper function 
******************************************************/
void shmem_set_lock( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_set_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_lock wrapper function 
******************************************************/
void SHMEM_SET_LOCK( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_set_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_lock wrapper function 
******************************************************/
void shmem_set_lock_( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_set_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_lock wrapper function 
******************************************************/
void shmem_set_lock__( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_set_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_clear_lock wrapper function 
******************************************************/
void shmem_clear_lock( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_lock wrapper function 
******************************************************/
void SHMEM_CLEAR_LOCK( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_lock wrapper function 
******************************************************/
void shmem_clear_lock_( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_lock wrapper function 
******************************************************/
void shmem_clear_lock__( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_test_lock wrapper function 
******************************************************/
int shmem_test_lock( long * lock)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_lock wrapper function 
******************************************************/
int SHMEM_TEST_LOCK( long * lock)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_lock wrapper function 
******************************************************/
int shmem_test_lock_( long * lock)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_lock wrapper function 
******************************************************/
int shmem_test_lock__( long * lock)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_lock( lock) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_lock( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/

#endif /* TAU_CATAMOUNT */


/******************************************************
***      shmem_clear_event wrapper function 
******************************************************/
void shmem_clear_event( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_event wrapper function 
******************************************************/
void SHMEM_CLEAR_EVENT( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_event wrapper function 
******************************************************/
void shmem_clear_event_( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_event wrapper function 
******************************************************/
void shmem_clear_event__( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_set_event wrapper function 
******************************************************/
void shmem_set_event( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_set_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_event wrapper function 
******************************************************/
void SHMEM_SET_EVENT( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_set_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_event wrapper function 
******************************************************/
void shmem_set_event_( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_set_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_event wrapper function 
******************************************************/
void shmem_set_event__( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_set_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_test_event wrapper function 
******************************************************/
int shmem_test_event( long * event)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_event wrapper function 
******************************************************/
int SHMEM_TEST_EVENT( long * event)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_event wrapper function 
******************************************************/
int shmem_test_event_( long * event)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_event wrapper function 
******************************************************/
int shmem_test_event__( long * event)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_wait_event wrapper function 
******************************************************/
void shmem_wait_event( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_event wrapper function 
******************************************************/
void SHMEM_WAIT_EVENT( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_event wrapper function 
******************************************************/
void shmem_wait_event_( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_event wrapper function 
******************************************************/
void shmem_wait_event__( long * event)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_event()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_event( event) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_event( event) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_wait wrapper function 
******************************************************/
void shmem_short_wait( short * var, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait wrapper function 
******************************************************/
void SHMEM_SHORT_WAIT( short * var, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait wrapper function 
******************************************************/
void shmem_short_wait_( short * var, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait wrapper function 
******************************************************/
void shmem_short_wait__( short * var, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_wait wrapper function 
******************************************************/
void shmem_int_wait( int * var, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait wrapper function 
******************************************************/
void SHMEM_INT_WAIT( int * var, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait wrapper function 
******************************************************/
void shmem_int_wait_( int * var, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait wrapper function 
******************************************************/
void shmem_int_wait__( int * var, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_wait wrapper function 
******************************************************/
void shmem_long_wait( long * var, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait wrapper function 
******************************************************/
void SHMEM_LONG_WAIT( long * var, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait wrapper function 
******************************************************/
void shmem_long_wait_( long * var, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait wrapper function 
******************************************************/
void shmem_long_wait__( long * var, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_wait wrapper function 
******************************************************/
void shmem_longlong_wait( long long * var, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait wrapper function 
******************************************************/
void SHMEM_LONGLONG_WAIT( long long * var, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait wrapper function 
******************************************************/
void shmem_longlong_wait_( long long * var, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait wrapper function 
******************************************************/
void shmem_longlong_wait__( long long * var, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_wait_until wrapper function 
******************************************************/
void shmem_short_wait_until( short * var, int cond, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait_until wrapper function 
******************************************************/
void SHMEM_SHORT_WAIT_UNTIL( short * var, int cond, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait_until wrapper function 
******************************************************/
void shmem_short_wait_until_( short * var, int cond, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait_until wrapper function 
******************************************************/
void shmem_short_wait_until__( short * var, int cond, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_wait_until wrapper function 
******************************************************/
void shmem_int_wait_until( int * var, int cond, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait_until wrapper function 
******************************************************/
void SHMEM_INT_WAIT_UNTIL( int * var, int cond, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait_until wrapper function 
******************************************************/
void shmem_int_wait_until_( int * var, int cond, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait_until wrapper function 
******************************************************/
void shmem_int_wait_until__( int * var, int cond, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_wait_until wrapper function 
******************************************************/
void shmem_long_wait_until( long * var, int cond, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait_until wrapper function 
******************************************************/
void SHMEM_LONG_WAIT_UNTIL( long * var, int cond, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait_until wrapper function 
******************************************************/
void shmem_long_wait_until_( long * var, int cond, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait_until wrapper function 
******************************************************/
void shmem_long_wait_until__( long * var, int cond, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_wait_until wrapper function 
******************************************************/
void shmem_longlong_wait_until( long long * var, int cond, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait_until wrapper function 
******************************************************/
void SHMEM_LONGLONG_WAIT_UNTIL( long long * var, int cond, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait_until wrapper function 
******************************************************/
void shmem_longlong_wait_until_( long long * var, int cond, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait_until wrapper function 
******************************************************/
void shmem_longlong_wait_until__( long long * var, int cond, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_sum_to_all wrapper function 
******************************************************/
void shmem_short_sum_to_all( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_SUM_TO_ALL( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all wrapper function 
******************************************************/
void shmem_short_sum_to_all_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all wrapper function 
******************************************************/
void shmem_short_sum_to_all__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_max_to_all wrapper function 
******************************************************/
void shmem_short_max_to_all( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_MAX_TO_ALL( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all wrapper function 
******************************************************/
void shmem_short_max_to_all_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all wrapper function 
******************************************************/
void shmem_short_max_to_all__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_min_to_all wrapper function 
******************************************************/
void shmem_short_min_to_all( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_MIN_TO_ALL( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all wrapper function 
******************************************************/
void shmem_short_min_to_all_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all wrapper function 
******************************************************/
void shmem_short_min_to_all__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_prod_to_all wrapper function 
******************************************************/
void shmem_short_prod_to_all( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_PROD_TO_ALL( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all wrapper function 
******************************************************/
void shmem_short_prod_to_all_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all wrapper function 
******************************************************/
void shmem_short_prod_to_all__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_and_to_all wrapper function 
******************************************************/
void shmem_short_and_to_all( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_AND_TO_ALL( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all wrapper function 
******************************************************/
void shmem_short_and_to_all_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all wrapper function 
******************************************************/
void shmem_short_and_to_all__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_or_to_all wrapper function 
******************************************************/
void shmem_short_or_to_all( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_OR_TO_ALL( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all wrapper function 
******************************************************/
void shmem_short_or_to_all_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all wrapper function 
******************************************************/
void shmem_short_or_to_all__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_xor_to_all wrapper function 
******************************************************/
void shmem_short_xor_to_all( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_XOR_TO_ALL( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all wrapper function 
******************************************************/
void shmem_short_xor_to_all_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all wrapper function 
******************************************************/
void shmem_short_xor_to_all__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_sum_to_all wrapper function 
******************************************************/
void shmem_int_sum_to_all( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all wrapper function 
******************************************************/
void SHMEM_INT_SUM_TO_ALL( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all wrapper function 
******************************************************/
void shmem_int_sum_to_all_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all wrapper function 
******************************************************/
void shmem_int_sum_to_all__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_max_to_all wrapper function 
******************************************************/
void shmem_int_max_to_all( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all wrapper function 
******************************************************/
void SHMEM_INT_MAX_TO_ALL( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all wrapper function 
******************************************************/
void shmem_int_max_to_all_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all wrapper function 
******************************************************/
void shmem_int_max_to_all__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_min_to_all wrapper function 
******************************************************/
void shmem_int_min_to_all( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all wrapper function 
******************************************************/
void SHMEM_INT_MIN_TO_ALL( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all wrapper function 
******************************************************/
void shmem_int_min_to_all_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all wrapper function 
******************************************************/
void shmem_int_min_to_all__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_prod_to_all wrapper function 
******************************************************/
void shmem_int_prod_to_all( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all wrapper function 
******************************************************/
void SHMEM_INT_PROD_TO_ALL( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all wrapper function 
******************************************************/
void shmem_int_prod_to_all_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all wrapper function 
******************************************************/
void shmem_int_prod_to_all__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_and_to_all wrapper function 
******************************************************/
void shmem_int_and_to_all( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all wrapper function 
******************************************************/
void SHMEM_INT_AND_TO_ALL( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all wrapper function 
******************************************************/
void shmem_int_and_to_all_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all wrapper function 
******************************************************/
void shmem_int_and_to_all__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_or_to_all wrapper function 
******************************************************/
void shmem_int_or_to_all( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all wrapper function 
******************************************************/
void SHMEM_INT_OR_TO_ALL( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all wrapper function 
******************************************************/
void shmem_int_or_to_all_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all wrapper function 
******************************************************/
void shmem_int_or_to_all__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_xor_to_all wrapper function 
******************************************************/
void shmem_int_xor_to_all( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all wrapper function 
******************************************************/
void SHMEM_INT_XOR_TO_ALL( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all wrapper function 
******************************************************/
void shmem_int_xor_to_all_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all wrapper function 
******************************************************/
void shmem_int_xor_to_all__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_sum_to_all wrapper function 
******************************************************/
void shmem_long_sum_to_all( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all wrapper function 
******************************************************/
void SHMEM_LONG_SUM_TO_ALL( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all wrapper function 
******************************************************/
void shmem_long_sum_to_all_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all wrapper function 
******************************************************/
void shmem_long_sum_to_all__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_max_to_all wrapper function 
******************************************************/
void shmem_long_max_to_all( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all wrapper function 
******************************************************/
void SHMEM_LONG_MAX_TO_ALL( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all wrapper function 
******************************************************/
void shmem_long_max_to_all_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all wrapper function 
******************************************************/
void shmem_long_max_to_all__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_min_to_all wrapper function 
******************************************************/
void shmem_long_min_to_all( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all wrapper function 
******************************************************/
void SHMEM_LONG_MIN_TO_ALL( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all wrapper function 
******************************************************/
void shmem_long_min_to_all_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all wrapper function 
******************************************************/
void shmem_long_min_to_all__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_prod_to_all wrapper function 
******************************************************/
void shmem_long_prod_to_all( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all wrapper function 
******************************************************/
void SHMEM_LONG_PROD_TO_ALL( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all wrapper function 
******************************************************/
void shmem_long_prod_to_all_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all wrapper function 
******************************************************/
void shmem_long_prod_to_all__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_and_to_all wrapper function 
******************************************************/
void shmem_long_and_to_all( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all wrapper function 
******************************************************/
void SHMEM_LONG_AND_TO_ALL( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all wrapper function 
******************************************************/
void shmem_long_and_to_all_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all wrapper function 
******************************************************/
void shmem_long_and_to_all__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_or_to_all wrapper function 
******************************************************/
void shmem_long_or_to_all( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all wrapper function 
******************************************************/
void SHMEM_LONG_OR_TO_ALL( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all wrapper function 
******************************************************/
void shmem_long_or_to_all_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all wrapper function 
******************************************************/
void shmem_long_or_to_all__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_xor_to_all wrapper function 
******************************************************/
void shmem_long_xor_to_all( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all wrapper function 
******************************************************/
void SHMEM_LONG_XOR_TO_ALL( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all wrapper function 
******************************************************/
void shmem_long_xor_to_all_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all wrapper function 
******************************************************/
void shmem_long_xor_to_all__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_sum_to_all wrapper function 
******************************************************/
void shmem_longlong_sum_to_all( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_SUM_TO_ALL( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all wrapper function 
******************************************************/
void shmem_longlong_sum_to_all_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all wrapper function 
******************************************************/
void shmem_longlong_sum_to_all__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_max_to_all wrapper function 
******************************************************/
void shmem_longlong_max_to_all( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_MAX_TO_ALL( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all wrapper function 
******************************************************/
void shmem_longlong_max_to_all_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all wrapper function 
******************************************************/
void shmem_longlong_max_to_all__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_min_to_all wrapper function 
******************************************************/
void shmem_longlong_min_to_all( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_MIN_TO_ALL( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all wrapper function 
******************************************************/
void shmem_longlong_min_to_all_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all wrapper function 
******************************************************/
void shmem_longlong_min_to_all__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_prod_to_all wrapper function 
******************************************************/
void shmem_longlong_prod_to_all( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_PROD_TO_ALL( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all wrapper function 
******************************************************/
void shmem_longlong_prod_to_all_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all wrapper function 
******************************************************/
void shmem_longlong_prod_to_all__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_and_to_all wrapper function 
******************************************************/
void shmem_longlong_and_to_all( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_AND_TO_ALL( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all wrapper function 
******************************************************/
void shmem_longlong_and_to_all_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all wrapper function 
******************************************************/
void shmem_longlong_and_to_all__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_or_to_all wrapper function 
******************************************************/
void shmem_longlong_or_to_all( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_OR_TO_ALL( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all wrapper function 
******************************************************/
void shmem_longlong_or_to_all_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all wrapper function 
******************************************************/
void shmem_longlong_or_to_all__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_xor_to_all wrapper function 
******************************************************/
void shmem_longlong_xor_to_all( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_XOR_TO_ALL( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all wrapper function 
******************************************************/
void shmem_longlong_xor_to_all_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all wrapper function 
******************************************************/
void shmem_longlong_xor_to_all__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_sum_to_all wrapper function 
******************************************************/
void shmem_float_sum_to_all( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_SUM_TO_ALL( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all wrapper function 
******************************************************/
void shmem_float_sum_to_all_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all wrapper function 
******************************************************/
void shmem_float_sum_to_all__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_max_to_all wrapper function 
******************************************************/
void shmem_float_max_to_all( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_MAX_TO_ALL( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all wrapper function 
******************************************************/
void shmem_float_max_to_all_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all wrapper function 
******************************************************/
void shmem_float_max_to_all__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_min_to_all wrapper function 
******************************************************/
void shmem_float_min_to_all( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_MIN_TO_ALL( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all wrapper function 
******************************************************/
void shmem_float_min_to_all_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all wrapper function 
******************************************************/
void shmem_float_min_to_all__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_prod_to_all wrapper function 
******************************************************/
void shmem_float_prod_to_all( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_PROD_TO_ALL( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all wrapper function 
******************************************************/
void shmem_float_prod_to_all_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all wrapper function 
******************************************************/
void shmem_float_prod_to_all__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_sum_to_all wrapper function 
******************************************************/
void shmem_double_sum_to_all( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_SUM_TO_ALL( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all wrapper function 
******************************************************/
void shmem_double_sum_to_all_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all wrapper function 
******************************************************/
void shmem_double_sum_to_all__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_max_to_all wrapper function 
******************************************************/
void shmem_double_max_to_all( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_MAX_TO_ALL( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all wrapper function 
******************************************************/
void shmem_double_max_to_all_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all wrapper function 
******************************************************/
void shmem_double_max_to_all__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_min_to_all wrapper function 
******************************************************/
void shmem_double_min_to_all( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_MIN_TO_ALL( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all wrapper function 
******************************************************/
void shmem_double_min_to_all_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all wrapper function 
******************************************************/
void shmem_double_min_to_all__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_prod_to_all wrapper function 
******************************************************/
void shmem_double_prod_to_all( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_PROD_TO_ALL( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all wrapper function 
******************************************************/
void shmem_double_prod_to_all_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all wrapper function 
******************************************************/
void shmem_double_prod_to_all__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_broadcast32 wrapper function 
******************************************************/
void shmem_broadcast32( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32 wrapper function 
******************************************************/
void SHMEM_BROADCAST32( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32 wrapper function 
******************************************************/
void shmem_broadcast32_( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32 wrapper function 
******************************************************/
void shmem_broadcast32__( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_broadcast64 wrapper function 
******************************************************/
void shmem_broadcast64( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64 wrapper function 
******************************************************/
void SHMEM_BROADCAST64( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64 wrapper function 
******************************************************/
void shmem_broadcast64_( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64 wrapper function 
******************************************************/
void shmem_broadcast64__( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_init wrapper function 
******************************************************/
void shmem_init( )
{

  TAU_PROFILE_TIMER(t, "shmem_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_init( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_init( ) ; 
#endif /* TAU_P_SHMEM */ 
  tau_totalnodes(1,shmem_n_pes());
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_init wrapper function 
******************************************************/
void SHMEM_INIT( )
{

  TAU_PROFILE_TIMER(t, "shmem_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_init( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_init( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_init wrapper function 
******************************************************/
void shmem_init_( )
{

  TAU_PROFILE_TIMER(t, "shmem_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_init( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_init( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_init wrapper function 
******************************************************/
void shmem_init__( )
{

  TAU_PROFILE_TIMER(t, "shmem_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_init( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_init( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_finalize wrapper function 
******************************************************/
void shmem_finalize( )
{

  TAU_PROFILE_TIMER(t, "shmem_finalize()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_finalize( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_finalize( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_finalize wrapper function 
******************************************************/
void SHMEM_FINALIZE( )
{

  TAU_PROFILE_TIMER(t, "shmem_finalize()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_finalize( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_finalize( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_finalize wrapper function 
******************************************************/
void shmem_finalize_( )
{

  TAU_PROFILE_TIMER(t, "shmem_finalize()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_finalize( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_finalize( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_finalize wrapper function 
******************************************************/
void shmem_finalize__( )
{

  TAU_PROFILE_TIMER(t, "shmem_finalize()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_finalize( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_finalize( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_n_pes wrapper function 
******************************************************/
int shmem_n_pes( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_n_pes()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_n_pes( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_n_pes( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_n_pes wrapper function 
******************************************************/
int SHMEM_N_PES( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_n_pes()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_n_pes( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_n_pes( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_n_pes wrapper function 
******************************************************/
int shmem_n_pes_( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_n_pes()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_n_pes( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_n_pes( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_n_pes wrapper function 
******************************************************/
int shmem_n_pes__( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_n_pes()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_n_pes( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_n_pes( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_my_pe wrapper function 
******************************************************/
int shmem_my_pe( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_my_pe()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_my_pe( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_my_pe( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_SET_NODE(retvalue);
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_my_pe wrapper function 
******************************************************/
int SHMEM_MY_PE( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_my_pe()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_my_pe( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_my_pe( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_SET_NODE(retvalue);
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_my_pe wrapper function 
******************************************************/
int shmem_my_pe_( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_my_pe()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_my_pe( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_my_pe( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_SET_NODE(retvalue);
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_my_pe wrapper function 
******************************************************/
int shmem_my_pe__( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_my_pe()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_my_pe( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_my_pe( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_SET_NODE(retvalue);
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fence wrapper function 
******************************************************/
void shmem_fence( )
{

  TAU_PROFILE_TIMER(t, "shmem_fence()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fence( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fence( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fence wrapper function 
******************************************************/
void SHMEM_FENCE( )
{

  TAU_PROFILE_TIMER(t, "shmem_fence()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fence( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fence( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fence wrapper function 
******************************************************/
void shmem_fence_( )
{

  TAU_PROFILE_TIMER(t, "shmem_fence()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fence( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fence( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fence wrapper function 
******************************************************/
void shmem_fence__( )
{

  TAU_PROFILE_TIMER(t, "shmem_fence()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fence( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fence( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


#ifndef TAU_CATAMOUNT
#ifndef TAU_CRAYCNL
/******************************************************
***      shmem_swap wrapper function 
******************************************************/
long shmem_swap( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap wrapper function 
******************************************************/
long SHMEM_SWAP( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap wrapper function 
******************************************************/
long shmem_swap_( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap wrapper function 
******************************************************/
long shmem_swap__( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_cswap wrapper function 
******************************************************/
long shmem_cswap( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_cswap wrapper function 
******************************************************/
long SHMEM_CSWAP( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_cswap wrapper function 
******************************************************/
long shmem_cswap_( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_cswap wrapper function 
******************************************************/
long shmem_cswap__( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_finc wrapper function 
******************************************************/
long shmem_finc( long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_finc wrapper function 
******************************************************/
long SHMEM_FINC( long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_finc wrapper function 
******************************************************/
long shmem_finc_( long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_finc wrapper function 
******************************************************/
long shmem_finc__( long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fadd wrapper function 
******************************************************/
long shmem_fadd( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_fadd wrapper function 
******************************************************/
long SHMEM_FADD( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_fadd wrapper function 
******************************************************/
long shmem_fadd_( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_fadd wrapper function 
******************************************************/
long shmem_fadd__( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_broadcast wrapper function 
******************************************************/
void shmem_broadcast( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast wrapper function 
******************************************************/
void SHMEM_BROADCAST( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast wrapper function 
******************************************************/
void shmem_broadcast_( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast wrapper function 
******************************************************/
void shmem_broadcast__( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get wrapper function 
******************************************************/
void shmem_get( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get wrapper function 
******************************************************/
void SHMEM_GET( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get wrapper function 
******************************************************/
void shmem_get_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get wrapper function 
******************************************************/
void shmem_get__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put wrapper function 
******************************************************/
void shmem_put( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put wrapper function 
******************************************************/
void SHMEM_PUT( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put wrapper function 
******************************************************/
void shmem_put_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put wrapper function 
******************************************************/
void shmem_put__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put_nb wrapper function 
******************************************************/
void shmem_put_nb( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put_nb wrapper function 
******************************************************/
void SHMEM_PUT_NB( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put_nb wrapper function 
******************************************************/
void shmem_put_nb_( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put_nb wrapper function 
******************************************************/
void shmem_put_nb__( void * trg, const void * src, size_t len, int pe, void ** transfer_handle)
{

  TAU_PROFILE_TIMER(t, "shmem_put_nb()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put_nb( trg, src, len, pe, transfer_handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put_nb( trg, src, len, pe, transfer_handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iget wrapper function 
******************************************************/
void shmem_iget( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget wrapper function 
******************************************************/
void SHMEM_IGET( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget wrapper function 
******************************************************/
void shmem_iget_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget wrapper function 
******************************************************/
void shmem_iget__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput wrapper function 
******************************************************/
void shmem_iput( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput wrapper function 
******************************************************/
void SHMEM_IPUT( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput wrapper function 
******************************************************/
void shmem_iput_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput wrapper function 
******************************************************/
void shmem_iput__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_wait wrapper function 
******************************************************/
void shmem_wait( long * var, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait wrapper function 
******************************************************/
void SHMEM_WAIT( long * var, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait wrapper function 
******************************************************/
void shmem_wait_( long * var, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait wrapper function 
******************************************************/
void shmem_wait__( long * var, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait( var, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait( var, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_wait_until wrapper function 
******************************************************/
void shmem_wait_until( long * var, int cond, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_until wrapper function 
******************************************************/
void SHMEM_WAIT_UNTIL( long * var, int cond, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_until wrapper function 
******************************************************/
void shmem_wait_until_( long * var, int cond, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_until wrapper function 
******************************************************/
void shmem_wait_until__( long * var, int cond, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_until( var, cond, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_until( var, cond, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

#endif /* TAU_CRAYCNL */
#endif /* TAU_CATAMOUNT: Cray has used #define for these symbols instead of 
          defining them independently */
