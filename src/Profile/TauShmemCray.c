#include <TAU.h>
#include <shmem.h>
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
***      shmem_get16_jw wrapper function 
******************************************************/
void shmem_get16_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get16_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get16_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get16_jw wrapper function 
******************************************************/
void SHMEM_GET16_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get16_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get16_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get16_jw wrapper function 
******************************************************/
void shmem_get16_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get16_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get16_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get16_jw wrapper function 
******************************************************/
void shmem_get16_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get16_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get16_jw( trg, src, len, pe) ; 
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
***      shmem_get32_jw wrapper function 
******************************************************/
void shmem_get32_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32_jw wrapper function 
******************************************************/
void SHMEM_GET32_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32_jw wrapper function 
******************************************************/
void shmem_get32_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32_jw wrapper function 
******************************************************/
void shmem_get32_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32_jw( trg, src, len, pe) ; 
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
***      shmem_get64_jw wrapper function 
******************************************************/
void shmem_get64_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64_jw wrapper function 
******************************************************/
void SHMEM_GET64_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64_jw wrapper function 
******************************************************/
void shmem_get64_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64_jw wrapper function 
******************************************************/
void shmem_get64_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64_jw( trg, src, len, pe) ; 
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
***      shmem_get128_jw wrapper function 
******************************************************/
void shmem_get128_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128_jw wrapper function 
******************************************************/
void SHMEM_GET128_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128_jw wrapper function 
******************************************************/
void shmem_get128_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128_jw wrapper function 
******************************************************/
void shmem_get128_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128_jw( trg, src, len, pe) ; 
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
***      shmem_getmem_jw wrapper function 
******************************************************/
void shmem_getmem_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem_jw wrapper function 
******************************************************/
void SHMEM_GETMEM_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem_jw wrapper function 
******************************************************/
void shmem_getmem_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem_jw wrapper function 
******************************************************/
void shmem_getmem_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem_jw( trg, src, len, pe) ; 
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
***      shmem_short_get_jw wrapper function 
******************************************************/
void shmem_short_get_jw( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get_jw wrapper function 
******************************************************/
void SHMEM_SHORT_GET_JW( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get_jw wrapper function 
******************************************************/
void shmem_short_get_jw_( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get_jw wrapper function 
******************************************************/
void shmem_short_get_jw__( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get_jw( trg, src, len, pe) ; 
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
***      shmem_int_get_jw wrapper function 
******************************************************/
void shmem_int_get_jw( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get_jw wrapper function 
******************************************************/
void SHMEM_INT_GET_JW( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get_jw wrapper function 
******************************************************/
void shmem_int_get_jw_( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get_jw wrapper function 
******************************************************/
void shmem_int_get_jw__( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get_jw( trg, src, len, pe) ; 
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
***      shmem_long_get_jw wrapper function 
******************************************************/
void shmem_long_get_jw( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get_jw wrapper function 
******************************************************/
void SHMEM_LONG_GET_JW( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get_jw wrapper function 
******************************************************/
void shmem_long_get_jw_( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get_jw wrapper function 
******************************************************/
void shmem_long_get_jw__( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get_jw( trg, src, len, pe) ; 
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
***      shmem_longlong_get_jw wrapper function 
******************************************************/
void shmem_longlong_get_jw( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_GET_JW( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get_jw wrapper function 
******************************************************/
void shmem_longlong_get_jw_( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get_jw wrapper function 
******************************************************/
void shmem_longlong_get_jw__( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get_jw( trg, src, len, pe) ; 
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
***      shmem_float_get_jw wrapper function 
******************************************************/
void shmem_float_get_jw( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_GET_JW( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get_jw wrapper function 
******************************************************/
void shmem_float_get_jw_( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get_jw wrapper function 
******************************************************/
void shmem_float_get_jw__( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get_jw( trg, src, len, pe) ; 
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
***      shmem_double_get_jw wrapper function 
******************************************************/
void shmem_double_get_jw( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_GET_JW( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get_jw wrapper function 
******************************************************/
void shmem_double_get_jw_( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get_jw wrapper function 
******************************************************/
void shmem_double_get_jw__( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_get wrapper function 
******************************************************/
void shmem_longdouble_get( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_GET( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get wrapper function 
******************************************************/
void shmem_longdouble_get_( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get wrapper function 
******************************************************/
void shmem_longdouble_get__( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_get_jw wrapper function 
******************************************************/
void shmem_longdouble_get_jw( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_GET_JW( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get_jw wrapper function 
******************************************************/
void shmem_longdouble_get_jw_( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get_jw wrapper function 
******************************************************/
void shmem_longdouble_get_jw__( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get_jw( trg, src, len, pe) ; 
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
***      shmem_put16_jw wrapper function 
******************************************************/
void shmem_put16_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16_jw wrapper function 
******************************************************/
void SHMEM_PUT16_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16_jw wrapper function 
******************************************************/
void shmem_put16_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put16_jw wrapper function 
******************************************************/
void shmem_put16_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put16_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put16_jw( trg, src, len, pe) ; 
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
***      shmem_put32_jw wrapper function 
******************************************************/
void shmem_put32_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32_jw wrapper function 
******************************************************/
void SHMEM_PUT32_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32_jw wrapper function 
******************************************************/
void shmem_put32_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32_jw wrapper function 
******************************************************/
void shmem_put32_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32_jw( trg, src, len, pe) ; 
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
***      shmem_put64_jw wrapper function 
******************************************************/
void shmem_put64_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64_jw wrapper function 
******************************************************/
void SHMEM_PUT64_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64_jw wrapper function 
******************************************************/
void shmem_put64_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64_jw wrapper function 
******************************************************/
void shmem_put64_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64_jw( trg, src, len, pe) ; 
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
***      shmem_put128_jw wrapper function 
******************************************************/
void shmem_put128_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128_jw wrapper function 
******************************************************/
void SHMEM_PUT128_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128_jw wrapper function 
******************************************************/
void shmem_put128_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128_jw wrapper function 
******************************************************/
void shmem_put128_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128_jw( trg, src, len, pe) ; 
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
***      shmem_putmem_jw wrapper function 
******************************************************/
void shmem_putmem_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem_jw wrapper function 
******************************************************/
void SHMEM_PUTMEM_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem_jw wrapper function 
******************************************************/
void shmem_putmem_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem_jw wrapper function 
******************************************************/
void shmem_putmem_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem_jw( trg, src, len, pe) ; 
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
***      shmem_short_put_jw wrapper function 
******************************************************/
void shmem_short_put_jw( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put_jw wrapper function 
******************************************************/
void SHMEM_SHORT_PUT_JW( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put_jw wrapper function 
******************************************************/
void shmem_short_put_jw_( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put_jw wrapper function 
******************************************************/
void shmem_short_put_jw__( short * trg, const short * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put_jw( trg, src, len, pe) ; 
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
***      shmem_int_put_jw wrapper function 
******************************************************/
void shmem_int_put_jw( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put_jw wrapper function 
******************************************************/
void SHMEM_INT_PUT_JW( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put_jw wrapper function 
******************************************************/
void shmem_int_put_jw_( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put_jw wrapper function 
******************************************************/
void shmem_int_put_jw__( int * trg, const int * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put_jw( trg, src, len, pe) ; 
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
***      shmem_long_put_jw wrapper function 
******************************************************/
void shmem_long_put_jw( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put_jw wrapper function 
******************************************************/
void SHMEM_LONG_PUT_JW( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put_jw wrapper function 
******************************************************/
void shmem_long_put_jw_( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put_jw wrapper function 
******************************************************/
void shmem_long_put_jw__( long * trg, const long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put_jw( trg, src, len, pe) ; 
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
***      shmem_longlong_put_jw wrapper function 
******************************************************/
void shmem_longlong_put_jw( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_PUT_JW( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put_jw wrapper function 
******************************************************/
void shmem_longlong_put_jw_( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put_jw wrapper function 
******************************************************/
void shmem_longlong_put_jw__( long long * trg, const long long * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put_jw( trg, src, len, pe) ; 
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
***      shmem_float_put_jw wrapper function 
******************************************************/
void shmem_float_put_jw( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_PUT_JW( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put_jw wrapper function 
******************************************************/
void shmem_float_put_jw_( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put_jw wrapper function 
******************************************************/
void shmem_float_put_jw__( float * trg, const float * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put_jw( trg, src, len, pe) ; 
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
***      shmem_double_put_jw wrapper function 
******************************************************/
void shmem_double_put_jw( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_PUT_JW( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put_jw wrapper function 
******************************************************/
void shmem_double_put_jw_( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put_jw wrapper function 
******************************************************/
void shmem_double_put_jw__( double * trg, const double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_put wrapper function 
******************************************************/
void shmem_longdouble_put( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_PUT( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put wrapper function 
******************************************************/
void shmem_longdouble_put_( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put wrapper function 
******************************************************/
void shmem_longdouble_put__( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_put_jw wrapper function 
******************************************************/
void shmem_longdouble_put_jw( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_PUT_JW( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put_jw wrapper function 
******************************************************/
void shmem_longdouble_put_jw_( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put_jw wrapper function 
******************************************************/
void shmem_longdouble_put_jw__( long double * trg, const long double * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put_jw( trg, src, len, pe) ; 
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
***      shmem_short_iget_jw wrapper function 
******************************************************/
void shmem_short_iget_jw( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget_jw wrapper function 
******************************************************/
void SHMEM_SHORT_IGET_JW( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget_jw wrapper function 
******************************************************/
void shmem_short_iget_jw_( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget_jw wrapper function 
******************************************************/
void shmem_short_iget_jw__( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_int_iget_jw wrapper function 
******************************************************/
void shmem_int_iget_jw( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget_jw wrapper function 
******************************************************/
void SHMEM_INT_IGET_JW( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget_jw wrapper function 
******************************************************/
void shmem_int_iget_jw_( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget_jw wrapper function 
******************************************************/
void shmem_int_iget_jw__( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_long_iget_jw wrapper function 
******************************************************/
void shmem_long_iget_jw( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget_jw wrapper function 
******************************************************/
void SHMEM_LONG_IGET_JW( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget_jw wrapper function 
******************************************************/
void shmem_long_iget_jw_( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget_jw wrapper function 
******************************************************/
void shmem_long_iget_jw__( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_longlong_iget_jw wrapper function 
******************************************************/
void shmem_longlong_iget_jw( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_IGET_JW( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget_jw wrapper function 
******************************************************/
void shmem_longlong_iget_jw_( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget_jw wrapper function 
******************************************************/
void shmem_longlong_iget_jw__( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_float_iget_jw wrapper function 
******************************************************/
void shmem_float_iget_jw( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_IGET_JW( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget_jw wrapper function 
******************************************************/
void shmem_float_iget_jw_( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget_jw wrapper function 
******************************************************/
void shmem_float_iget_jw__( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_double_iget_jw wrapper function 
******************************************************/
void shmem_double_iget_jw( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_IGET_JW( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget_jw wrapper function 
******************************************************/
void shmem_double_iget_jw_( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget_jw wrapper function 
******************************************************/
void shmem_double_iget_jw__( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_iget wrapper function 
******************************************************/
void shmem_longdouble_iget( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_IGET( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget wrapper function 
******************************************************/
void shmem_longdouble_iget_( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget wrapper function 
******************************************************/
void shmem_longdouble_iget__( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_iget_jw wrapper function 
******************************************************/
void shmem_longdouble_iget_jw( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_IGET_JW( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget_jw wrapper function 
******************************************************/
void shmem_longdouble_iget_jw_( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget_jw wrapper function 
******************************************************/
void shmem_longdouble_iget_jw__( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iget16_jw wrapper function 
******************************************************/
void shmem_iget16_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget16_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget16_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget16_jw wrapper function 
******************************************************/
void SHMEM_IGET16_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget16_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget16_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget16_jw wrapper function 
******************************************************/
void shmem_iget16_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget16_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget16_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget16_jw wrapper function 
******************************************************/
void shmem_iget16_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget16_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget16_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iget32_jw wrapper function 
******************************************************/
void shmem_iget32_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32_jw wrapper function 
******************************************************/
void SHMEM_IGET32_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32_jw wrapper function 
******************************************************/
void shmem_iget32_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32_jw wrapper function 
******************************************************/
void shmem_iget32_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iget64_jw wrapper function 
******************************************************/
void shmem_iget64_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64_jw wrapper function 
******************************************************/
void SHMEM_IGET64_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64_jw wrapper function 
******************************************************/
void shmem_iget64_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64_jw wrapper function 
******************************************************/
void shmem_iget64_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iget128_jw wrapper function 
******************************************************/
void shmem_iget128_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128_jw wrapper function 
******************************************************/
void SHMEM_IGET128_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128_jw wrapper function 
******************************************************/
void shmem_iget128_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128_jw wrapper function 
******************************************************/
void shmem_iget128_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_short_iput_jw wrapper function 
******************************************************/
void shmem_short_iput_jw( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput_jw wrapper function 
******************************************************/
void SHMEM_SHORT_IPUT_JW( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput_jw wrapper function 
******************************************************/
void shmem_short_iput_jw_( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput_jw wrapper function 
******************************************************/
void shmem_short_iput_jw__( short * trg, const short * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_int_iput_jw wrapper function 
******************************************************/
void shmem_int_iput_jw( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput_jw wrapper function 
******************************************************/
void SHMEM_INT_IPUT_JW( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput_jw wrapper function 
******************************************************/
void shmem_int_iput_jw_( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput_jw wrapper function 
******************************************************/
void shmem_int_iput_jw__( int * trg, const int * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_long_iput_jw wrapper function 
******************************************************/
void shmem_long_iput_jw( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput_jw wrapper function 
******************************************************/
void SHMEM_LONG_IPUT_JW( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput_jw wrapper function 
******************************************************/
void shmem_long_iput_jw_( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput_jw wrapper function 
******************************************************/
void shmem_long_iput_jw__( long * trg, const long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_longlong_iput_jw wrapper function 
******************************************************/
void shmem_longlong_iput_jw( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_IPUT_JW( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput_jw wrapper function 
******************************************************/
void shmem_longlong_iput_jw_( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput_jw wrapper function 
******************************************************/
void shmem_longlong_iput_jw__( long long * trg, const long long * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_float_iput_jw wrapper function 
******************************************************/
void shmem_float_iput_jw( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_IPUT_JW( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput_jw wrapper function 
******************************************************/
void shmem_float_iput_jw_( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput_jw wrapper function 
******************************************************/
void shmem_float_iput_jw__( float * trg, const float * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_double_iput_jw wrapper function 
******************************************************/
void shmem_double_iput_jw( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_IPUT_JW( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput_jw wrapper function 
******************************************************/
void shmem_double_iput_jw_( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput_jw wrapper function 
******************************************************/
void shmem_double_iput_jw__( double * trg, const double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_iput wrapper function 
******************************************************/
void shmem_longdouble_iput( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_IPUT( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput wrapper function 
******************************************************/
void shmem_longdouble_iput_( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput wrapper function 
******************************************************/
void shmem_longdouble_iput__( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_iput_jw wrapper function 
******************************************************/
void shmem_longdouble_iput_jw( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_IPUT_JW( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput_jw wrapper function 
******************************************************/
void shmem_longdouble_iput_jw_( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput_jw wrapper function 
******************************************************/
void shmem_longdouble_iput_jw__( long double * trg, const long double * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iput16_jw wrapper function 
******************************************************/
void shmem_iput16_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput16_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput16_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput16_jw wrapper function 
******************************************************/
void SHMEM_IPUT16_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput16_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput16_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput16_jw wrapper function 
******************************************************/
void shmem_iput16_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput16_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput16_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput16_jw wrapper function 
******************************************************/
void shmem_iput16_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput16_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput16_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput16_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iput32_jw wrapper function 
******************************************************/
void shmem_iput32_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32_jw wrapper function 
******************************************************/
void SHMEM_IPUT32_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32_jw wrapper function 
******************************************************/
void shmem_iput32_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32_jw wrapper function 
******************************************************/
void shmem_iput32_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iput64_jw wrapper function 
******************************************************/
void shmem_iput64_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64_jw wrapper function 
******************************************************/
void SHMEM_IPUT64_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64_jw wrapper function 
******************************************************/
void shmem_iput64_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64_jw wrapper function 
******************************************************/
void shmem_iput64_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iput128_jw wrapper function 
******************************************************/
void shmem_iput128_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128_jw wrapper function 
******************************************************/
void SHMEM_IPUT128_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128_jw wrapper function 
******************************************************/
void shmem_iput128_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128_jw wrapper function 
******************************************************/
void shmem_iput128_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget32 wrapper function 
******************************************************/
void shmem_ixget32( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32 wrapper function 
******************************************************/
void SHMEM_IXGET32( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32 wrapper function 
******************************************************/
void shmem_ixget32_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32 wrapper function 
******************************************************/
void shmem_ixget32__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget32_jw wrapper function 
******************************************************/
void shmem_ixget32_jw( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32_jw wrapper function 
******************************************************/
void SHMEM_IXGET32_JW( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32_jw wrapper function 
******************************************************/
void shmem_ixget32_jw_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32_jw wrapper function 
******************************************************/
void shmem_ixget32_jw__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix32get32 wrapper function 
******************************************************/
void shmem_ix32get32( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get32 wrapper function 
******************************************************/
void SHMEM_IX32GET32( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get32 wrapper function 
******************************************************/
void shmem_ix32get32_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get32 wrapper function 
******************************************************/
void shmem_ix32get32__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix32get32_jw wrapper function 
******************************************************/
void shmem_ix32get32_jw( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get32_jw wrapper function 
******************************************************/
void SHMEM_IX32GET32_JW( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get32_jw wrapper function 
******************************************************/
void shmem_ix32get32_jw_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get32_jw wrapper function 
******************************************************/
void shmem_ix32get32_jw__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget64 wrapper function 
******************************************************/
void shmem_ixget64( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64 wrapper function 
******************************************************/
void SHMEM_IXGET64( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64 wrapper function 
******************************************************/
void shmem_ixget64_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64 wrapper function 
******************************************************/
void shmem_ixget64__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget64_jw wrapper function 
******************************************************/
void shmem_ixget64_jw( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64_jw wrapper function 
******************************************************/
void SHMEM_IXGET64_JW( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64_jw wrapper function 
******************************************************/
void shmem_ixget64_jw_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64_jw wrapper function 
******************************************************/
void shmem_ixget64_jw__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix64get64 wrapper function 
******************************************************/
void shmem_ix64get64( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get64 wrapper function 
******************************************************/
void SHMEM_IX64GET64( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get64 wrapper function 
******************************************************/
void shmem_ix64get64_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get64 wrapper function 
******************************************************/
void shmem_ix64get64__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix64get64_jw wrapper function 
******************************************************/
void shmem_ix64get64_jw( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get64_jw wrapper function 
******************************************************/
void SHMEM_IX64GET64_JW( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get64_jw wrapper function 
******************************************************/
void shmem_ix64get64_jw_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get64_jw wrapper function 
******************************************************/
void shmem_ix64get64_jw__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix32get64 wrapper function 
******************************************************/
void shmem_ix32get64( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get64 wrapper function 
******************************************************/
void SHMEM_IX32GET64( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get64 wrapper function 
******************************************************/
void shmem_ix32get64_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get64 wrapper function 
******************************************************/
void shmem_ix32get64__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix32get64_jw wrapper function 
******************************************************/
void shmem_ix32get64_jw( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get64_jw wrapper function 
******************************************************/
void SHMEM_IX32GET64_JW( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get64_jw wrapper function 
******************************************************/
void shmem_ix32get64_jw_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32get64_jw wrapper function 
******************************************************/
void shmem_ix32get64_jw__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32get64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32get64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32get64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix64get32 wrapper function 
******************************************************/
void shmem_ix64get32( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get32 wrapper function 
******************************************************/
void SHMEM_IX64GET32( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get32 wrapper function 
******************************************************/
void shmem_ix64get32_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get32 wrapper function 
******************************************************/
void shmem_ix64get32__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix64get32_jw wrapper function 
******************************************************/
void shmem_ix64get32_jw( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get32_jw wrapper function 
******************************************************/
void SHMEM_IX64GET32_JW( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get32_jw wrapper function 
******************************************************/
void shmem_ix64get32_jw_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64get32_jw wrapper function 
******************************************************/
void shmem_ix64get32_jw__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64get32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64get32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64get32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput32 wrapper function 
******************************************************/
void shmem_ixput32( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32 wrapper function 
******************************************************/
void SHMEM_IXPUT32( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32 wrapper function 
******************************************************/
void shmem_ixput32_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32 wrapper function 
******************************************************/
void shmem_ixput32__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput32_jw wrapper function 
******************************************************/
void shmem_ixput32_jw( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32_jw wrapper function 
******************************************************/
void SHMEM_IXPUT32_JW( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32_jw wrapper function 
******************************************************/
void shmem_ixput32_jw_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32_jw wrapper function 
******************************************************/
void shmem_ixput32_jw__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix32put32 wrapper function 
******************************************************/
void shmem_ix32put32( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put32 wrapper function 
******************************************************/
void SHMEM_IX32PUT32( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put32 wrapper function 
******************************************************/
void shmem_ix32put32_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put32 wrapper function 
******************************************************/
void shmem_ix32put32__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix32put32_jw wrapper function 
******************************************************/
void shmem_ix32put32_jw( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put32_jw wrapper function 
******************************************************/
void SHMEM_IX32PUT32_JW( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put32_jw wrapper function 
******************************************************/
void shmem_ix32put32_jw_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put32_jw wrapper function 
******************************************************/
void shmem_ix32put32_jw__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput64 wrapper function 
******************************************************/
void shmem_ixput64( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64 wrapper function 
******************************************************/
void SHMEM_IXPUT64( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64 wrapper function 
******************************************************/
void shmem_ixput64_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64 wrapper function 
******************************************************/
void shmem_ixput64__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput64_jw wrapper function 
******************************************************/
void shmem_ixput64_jw( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64_jw wrapper function 
******************************************************/
void SHMEM_IXPUT64_JW( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64_jw wrapper function 
******************************************************/
void shmem_ixput64_jw_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64_jw wrapper function 
******************************************************/
void shmem_ixput64_jw__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix64put64 wrapper function 
******************************************************/
void shmem_ix64put64( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put64 wrapper function 
******************************************************/
void SHMEM_IX64PUT64( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put64 wrapper function 
******************************************************/
void shmem_ix64put64_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put64 wrapper function 
******************************************************/
void shmem_ix64put64__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix64put64_jw wrapper function 
******************************************************/
void shmem_ix64put64_jw( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put64_jw wrapper function 
******************************************************/
void SHMEM_IX64PUT64_JW( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put64_jw wrapper function 
******************************************************/
void shmem_ix64put64_jw_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put64_jw wrapper function 
******************************************************/
void shmem_ix64put64_jw__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix32put64 wrapper function 
******************************************************/
void shmem_ix32put64( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put64 wrapper function 
******************************************************/
void SHMEM_IX32PUT64( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put64 wrapper function 
******************************************************/
void shmem_ix32put64_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put64 wrapper function 
******************************************************/
void shmem_ix32put64__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put64( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put64( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix32put64_jw wrapper function 
******************************************************/
void shmem_ix32put64_jw( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put64_jw wrapper function 
******************************************************/
void SHMEM_IX32PUT64_JW( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put64_jw wrapper function 
******************************************************/
void shmem_ix32put64_jw_( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix32put64_jw wrapper function 
******************************************************/
void shmem_ix32put64_jw__( void * trg, const void * src, int * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix32put64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix32put64_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix32put64_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix64put32 wrapper function 
******************************************************/
void shmem_ix64put32( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put32 wrapper function 
******************************************************/
void SHMEM_IX64PUT32( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put32 wrapper function 
******************************************************/
void shmem_ix64put32_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put32 wrapper function 
******************************************************/
void shmem_ix64put32__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put32( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put32( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ix64put32_jw wrapper function 
******************************************************/
void shmem_ix64put32_jw( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put32_jw wrapper function 
******************************************************/
void SHMEM_IX64PUT32_JW( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put32_jw wrapper function 
******************************************************/
void shmem_ix64put32_jw_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put32_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ix64put32_jw wrapper function 
******************************************************/
void shmem_ix64put32_jw__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ix64put32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ix64put32_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ix64put32_jw( trg, src, idx, len, pe) ; 
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
***      shmem_char_g_jw wrapper function 
******************************************************/
char shmem_char_g_jw( const char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_g_jw wrapper function 
******************************************************/
char SHMEM_CHAR_G_JW( const char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_g_jw wrapper function 
******************************************************/
char shmem_char_g_jw_( const char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_g_jw wrapper function 
******************************************************/
char shmem_char_g_jw__( const char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_g_jw( addr, pe) ; 
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
***      shmem_short_g_jw wrapper function 
******************************************************/
short shmem_short_g_jw( const short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_g_jw wrapper function 
******************************************************/
short SHMEM_SHORT_G_JW( const short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_g_jw wrapper function 
******************************************************/
short shmem_short_g_jw_( const short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_g_jw wrapper function 
******************************************************/
short shmem_short_g_jw__( const short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_g_jw( addr, pe) ; 
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
***      shmem_int_g_jw wrapper function 
******************************************************/
int shmem_int_g_jw( const int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_g_jw wrapper function 
******************************************************/
int SHMEM_INT_G_JW( const int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_g_jw wrapper function 
******************************************************/
int shmem_int_g_jw_( const int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_g_jw wrapper function 
******************************************************/
int shmem_int_g_jw__( const int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_g_jw( addr, pe) ; 
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
***      shmem_long_g_jw wrapper function 
******************************************************/
long shmem_long_g_jw( const long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_g_jw wrapper function 
******************************************************/
long SHMEM_LONG_G_JW( const long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_g_jw wrapper function 
******************************************************/
long shmem_long_g_jw_( const long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_g_jw wrapper function 
******************************************************/
long shmem_long_g_jw__( const long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_g_jw( addr, pe) ; 
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
***      shmem_longlong_g_jw wrapper function 
******************************************************/
long long shmem_longlong_g_jw( const long long * addr, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_g_jw wrapper function 
******************************************************/
long long SHMEM_LONGLONG_G_JW( const long long * addr, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_g_jw wrapper function 
******************************************************/
long long shmem_longlong_g_jw_( const long long * addr, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_g_jw wrapper function 
******************************************************/
long long shmem_longlong_g_jw__( const long long * addr, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_g_jw( addr, pe) ; 
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
***      shmem_float_g_jw wrapper function 
******************************************************/
float shmem_float_g_jw( const float * addr, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_g_jw wrapper function 
******************************************************/
float SHMEM_FLOAT_G_JW( const float * addr, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_g_jw wrapper function 
******************************************************/
float shmem_float_g_jw_( const float * addr, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_g_jw wrapper function 
******************************************************/
float shmem_float_g_jw__( const float * addr, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_g_jw( addr, pe) ; 
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
***      shmem_double_g_jw wrapper function 
******************************************************/
double shmem_double_g_jw( const double * addr, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_g_jw wrapper function 
******************************************************/
double SHMEM_DOUBLE_G_JW( const double * addr, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_g_jw wrapper function 
******************************************************/
double shmem_double_g_jw_( const double * addr, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_g_jw wrapper function 
******************************************************/
double shmem_double_g_jw__( const double * addr, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_g wrapper function 
******************************************************/
long double shmem_longdouble_g( const long double * addr, int pe)
{
  long double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longdouble_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longdouble_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longdouble_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longdouble_g wrapper function 
******************************************************/
long double SHMEM_LONGDOUBLE_G( const long double * addr, int pe)
{
  long double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longdouble_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longdouble_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longdouble_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longdouble_g wrapper function 
******************************************************/
long double shmem_longdouble_g_( const long double * addr, int pe)
{
  long double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longdouble_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longdouble_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longdouble_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longdouble_g wrapper function 
******************************************************/
long double shmem_longdouble_g__( const long double * addr, int pe)
{
  long double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longdouble_g()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longdouble_g( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longdouble_g( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_g_jw wrapper function 
******************************************************/
long double shmem_longdouble_g_jw( const long double * addr, int pe)
{
  long double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longdouble_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longdouble_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longdouble_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longdouble_g_jw wrapper function 
******************************************************/
long double SHMEM_LONGDOUBLE_G_JW( const long double * addr, int pe)
{
  long double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longdouble_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longdouble_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longdouble_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longdouble_g_jw wrapper function 
******************************************************/
long double shmem_longdouble_g_jw_( const long double * addr, int pe)
{
  long double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longdouble_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longdouble_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longdouble_g_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longdouble_g_jw wrapper function 
******************************************************/
long double shmem_longdouble_g_jw__( const long double * addr, int pe)
{
  long double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longdouble_g_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longdouble_g_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longdouble_g_jw( addr, pe) ; 
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
***      shmem_char_p_jw wrapper function 
******************************************************/
void shmem_char_p_jw( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_p_jw wrapper function 
******************************************************/
void SHMEM_CHAR_P_JW( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_p_jw wrapper function 
******************************************************/
void shmem_char_p_jw_( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_p_jw wrapper function 
******************************************************/
void shmem_char_p_jw__( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_p_jw( addr, value, pe) ; 
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
***      shmem_short_p_jw wrapper function 
******************************************************/
void shmem_short_p_jw( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_p_jw wrapper function 
******************************************************/
void SHMEM_SHORT_P_JW( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_p_jw wrapper function 
******************************************************/
void shmem_short_p_jw_( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_p_jw wrapper function 
******************************************************/
void shmem_short_p_jw__( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_p_jw( addr, value, pe) ; 
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
***      shmem_int_p_jw wrapper function 
******************************************************/
void shmem_int_p_jw( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_p_jw wrapper function 
******************************************************/
void SHMEM_INT_P_JW( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_p_jw wrapper function 
******************************************************/
void shmem_int_p_jw_( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_p_jw wrapper function 
******************************************************/
void shmem_int_p_jw__( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_p_jw( addr, value, pe) ; 
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
***      shmem_long_p_jw wrapper function 
******************************************************/
void shmem_long_p_jw( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_p_jw wrapper function 
******************************************************/
void SHMEM_LONG_P_JW( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_p_jw wrapper function 
******************************************************/
void shmem_long_p_jw_( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_p_jw wrapper function 
******************************************************/
void shmem_long_p_jw__( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_p_jw( addr, value, pe) ; 
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
***      shmem_longlong_p_jw wrapper function 
******************************************************/
void shmem_longlong_p_jw( long long * addr, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_p_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_P_JW( long long * addr, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_p_jw wrapper function 
******************************************************/
void shmem_longlong_p_jw_( long long * addr, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_p_jw wrapper function 
******************************************************/
void shmem_longlong_p_jw__( long long * addr, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_p_jw( addr, value, pe) ; 
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
***      shmem_float_p_jw wrapper function 
******************************************************/
void shmem_float_p_jw( float * addr, float value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_p_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_P_JW( float * addr, float value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_p_jw wrapper function 
******************************************************/
void shmem_float_p_jw_( float * addr, float value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_p_jw wrapper function 
******************************************************/
void shmem_float_p_jw__( float * addr, float value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_p_jw( addr, value, pe) ; 
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
***      shmem_double_p_jw wrapper function 
******************************************************/
void shmem_double_p_jw( double * addr, double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_p_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_P_JW( double * addr, double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_p_jw wrapper function 
******************************************************/
void shmem_double_p_jw_( double * addr, double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_p_jw wrapper function 
******************************************************/
void shmem_double_p_jw__( double * addr, double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_p wrapper function 
******************************************************/
void shmem_longdouble_p( long double * addr, long double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_p wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_P( long double * addr, long double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_p wrapper function 
******************************************************/
void shmem_longdouble_p_( long double * addr, long double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_p wrapper function 
******************************************************/
void shmem_longdouble_p__( long double * addr, long double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_p()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_p( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_p( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_p_jw wrapper function 
******************************************************/
void shmem_longdouble_p_jw( long double * addr, long double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_p_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_P_JW( long double * addr, long double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_p_jw wrapper function 
******************************************************/
void shmem_longdouble_p_jw_( long double * addr, long double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_p_jw wrapper function 
******************************************************/
void shmem_longdouble_p_jw__( long double * addr, long double value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_p_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_p_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_p_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_swap wrapper function 
******************************************************/
char shmem_char_swap( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_swap wrapper function 
******************************************************/
char SHMEM_CHAR_SWAP( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_swap wrapper function 
******************************************************/
char shmem_char_swap_( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_swap wrapper function 
******************************************************/
char shmem_char_swap__( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_swap_jw wrapper function 
******************************************************/
char shmem_char_swap_jw( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_swap_jw wrapper function 
******************************************************/
char SHMEM_CHAR_SWAP_JW( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_swap_jw wrapper function 
******************************************************/
char shmem_char_swap_jw_( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_swap_jw wrapper function 
******************************************************/
char shmem_char_swap_jw__( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_swap wrapper function 
******************************************************/
short shmem_short_swap( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap wrapper function 
******************************************************/
short SHMEM_SHORT_SWAP( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap wrapper function 
******************************************************/
short shmem_short_swap_( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap wrapper function 
******************************************************/
short shmem_short_swap__( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_swap_jw wrapper function 
******************************************************/
short shmem_short_swap_jw( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap_jw wrapper function 
******************************************************/
short SHMEM_SHORT_SWAP_JW( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap_jw wrapper function 
******************************************************/
short shmem_short_swap_jw_( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap_jw wrapper function 
******************************************************/
short shmem_short_swap_jw__( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
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
***      shmem_int_swap_jw wrapper function 
******************************************************/
int shmem_int_swap_jw( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap_jw wrapper function 
******************************************************/
int SHMEM_INT_SWAP_JW( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap_jw wrapper function 
******************************************************/
int shmem_int_swap_jw_( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap_jw wrapper function 
******************************************************/
int shmem_int_swap_jw__( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap_jw( addr, value, pe) ; 
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


/******************************************************
***      shmem_long_swap_jw wrapper function 
******************************************************/
long shmem_long_swap_jw( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap_jw wrapper function 
******************************************************/
long SHMEM_LONG_SWAP_JW( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap_jw wrapper function 
******************************************************/
long shmem_long_swap_jw_( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap_jw wrapper function 
******************************************************/
long shmem_long_swap_jw__( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_swap wrapper function 
******************************************************/
long long shmem_longlong_swap( long long * addr, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap wrapper function 
******************************************************/
long long SHMEM_LONGLONG_SWAP( long long * addr, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap wrapper function 
******************************************************/
long long shmem_longlong_swap_( long long * addr, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap wrapper function 
******************************************************/
long long shmem_longlong_swap__( long long * addr, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_swap_jw wrapper function 
******************************************************/
long long shmem_longlong_swap_jw( long long * addr, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap_jw wrapper function 
******************************************************/
long long SHMEM_LONGLONG_SWAP_JW( long long * addr, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap_jw wrapper function 
******************************************************/
long long shmem_longlong_swap_jw_( long long * addr, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap_jw wrapper function 
******************************************************/
long long shmem_longlong_swap_jw__( long long * addr, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


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
***      shmem_float_swap_jw wrapper function 
******************************************************/
float shmem_float_swap_jw( float * addr, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap_jw wrapper function 
******************************************************/
float SHMEM_FLOAT_SWAP_JW( float * addr, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap_jw wrapper function 
******************************************************/
float shmem_float_swap_jw_( float * addr, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap_jw wrapper function 
******************************************************/
float shmem_float_swap_jw__( float * addr, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap_jw( addr, value, pe) ; 
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
***      shmem_double_swap_jw wrapper function 
******************************************************/
double shmem_double_swap_jw( double * addr, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap_jw wrapper function 
******************************************************/
double SHMEM_DOUBLE_SWAP_JW( double * addr, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap_jw wrapper function 
******************************************************/
double shmem_double_swap_jw_( double * addr, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap_jw wrapper function 
******************************************************/
double shmem_double_swap_jw__( double * addr, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_mswap wrapper function 
******************************************************/
char shmem_char_mswap( char * addr, char mask, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_mswap wrapper function 
******************************************************/
char SHMEM_CHAR_MSWAP( char * addr, char mask, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_mswap wrapper function 
******************************************************/
char shmem_char_mswap_( char * addr, char mask, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_mswap wrapper function 
******************************************************/
char shmem_char_mswap__( char * addr, char mask, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_mswap_jw wrapper function 
******************************************************/
char shmem_char_mswap_jw( char * addr, char mask, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_mswap_jw wrapper function 
******************************************************/
char SHMEM_CHAR_MSWAP_JW( char * addr, char mask, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_mswap_jw wrapper function 
******************************************************/
char shmem_char_mswap_jw_( char * addr, char mask, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_mswap_jw wrapper function 
******************************************************/
char shmem_char_mswap_jw__( char * addr, char mask, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_mswap wrapper function 
******************************************************/
short shmem_short_mswap( short * addr, short mask, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_mswap wrapper function 
******************************************************/
short SHMEM_SHORT_MSWAP( short * addr, short mask, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_mswap wrapper function 
******************************************************/
short shmem_short_mswap_( short * addr, short mask, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_mswap wrapper function 
******************************************************/
short shmem_short_mswap__( short * addr, short mask, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_mswap_jw wrapper function 
******************************************************/
short shmem_short_mswap_jw( short * addr, short mask, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_mswap_jw wrapper function 
******************************************************/
short SHMEM_SHORT_MSWAP_JW( short * addr, short mask, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_mswap_jw wrapper function 
******************************************************/
short shmem_short_mswap_jw_( short * addr, short mask, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_mswap_jw wrapper function 
******************************************************/
short shmem_short_mswap_jw__( short * addr, short mask, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_mswap wrapper function 
******************************************************/
int shmem_int_mswap( int * addr, int mask, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_mswap wrapper function 
******************************************************/
int SHMEM_INT_MSWAP( int * addr, int mask, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_mswap wrapper function 
******************************************************/
int shmem_int_mswap_( int * addr, int mask, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_mswap wrapper function 
******************************************************/
int shmem_int_mswap__( int * addr, int mask, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_mswap_jw wrapper function 
******************************************************/
int shmem_int_mswap_jw( int * addr, int mask, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_mswap_jw wrapper function 
******************************************************/
int SHMEM_INT_MSWAP_JW( int * addr, int mask, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_mswap_jw wrapper function 
******************************************************/
int shmem_int_mswap_jw_( int * addr, int mask, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_mswap_jw wrapper function 
******************************************************/
int shmem_int_mswap_jw__( int * addr, int mask, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_mswap wrapper function 
******************************************************/
long shmem_long_mswap( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_mswap wrapper function 
******************************************************/
long SHMEM_LONG_MSWAP( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_mswap wrapper function 
******************************************************/
long shmem_long_mswap_( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_mswap wrapper function 
******************************************************/
long shmem_long_mswap__( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_mswap_jw wrapper function 
******************************************************/
long shmem_long_mswap_jw( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_mswap_jw wrapper function 
******************************************************/
long SHMEM_LONG_MSWAP_JW( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_mswap_jw wrapper function 
******************************************************/
long shmem_long_mswap_jw_( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_mswap_jw wrapper function 
******************************************************/
long shmem_long_mswap_jw__( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_mswap wrapper function 
******************************************************/
long long shmem_longlong_mswap( long long * addr, long long mask, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_mswap wrapper function 
******************************************************/
long long SHMEM_LONGLONG_MSWAP( long long * addr, long long mask, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_mswap wrapper function 
******************************************************/
long long shmem_longlong_mswap_( long long * addr, long long mask, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_mswap wrapper function 
******************************************************/
long long shmem_longlong_mswap__( long long * addr, long long mask, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_mswap_jw wrapper function 
******************************************************/
long long shmem_longlong_mswap_jw( long long * addr, long long mask, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_mswap_jw wrapper function 
******************************************************/
long long SHMEM_LONGLONG_MSWAP_JW( long long * addr, long long mask, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_mswap_jw wrapper function 
******************************************************/
long long shmem_longlong_mswap_jw_( long long * addr, long long mask, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_mswap_jw wrapper function 
******************************************************/
long long shmem_longlong_mswap_jw__( long long * addr, long long mask, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_cswap wrapper function 
******************************************************/
char shmem_char_cswap( char * addr, char match, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_cswap wrapper function 
******************************************************/
char SHMEM_CHAR_CSWAP( char * addr, char match, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_cswap wrapper function 
******************************************************/
char shmem_char_cswap_( char * addr, char match, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_cswap wrapper function 
******************************************************/
char shmem_char_cswap__( char * addr, char match, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_cswap_jw wrapper function 
******************************************************/
char shmem_char_cswap_jw( char * addr, char match, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_cswap_jw wrapper function 
******************************************************/
char SHMEM_CHAR_CSWAP_JW( char * addr, char match, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_cswap_jw wrapper function 
******************************************************/
char shmem_char_cswap_jw_( char * addr, char match, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_cswap_jw wrapper function 
******************************************************/
char shmem_char_cswap_jw__( char * addr, char match, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_cswap wrapper function 
******************************************************/
short shmem_short_cswap( short * addr, short match, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap wrapper function 
******************************************************/
short SHMEM_SHORT_CSWAP( short * addr, short match, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap wrapper function 
******************************************************/
short shmem_short_cswap_( short * addr, short match, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap wrapper function 
******************************************************/
short shmem_short_cswap__( short * addr, short match, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_cswap_jw wrapper function 
******************************************************/
short shmem_short_cswap_jw( short * addr, short match, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap_jw wrapper function 
******************************************************/
short SHMEM_SHORT_CSWAP_JW( short * addr, short match, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap_jw wrapper function 
******************************************************/
short shmem_short_cswap_jw_( short * addr, short match, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap_jw wrapper function 
******************************************************/
short shmem_short_cswap_jw__( short * addr, short match, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap_jw( addr, match, value, pe) ; 
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
***      shmem_int_cswap_jw wrapper function 
******************************************************/
int shmem_int_cswap_jw( int * addr, int match, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap_jw wrapper function 
******************************************************/
int SHMEM_INT_CSWAP_JW( int * addr, int match, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap_jw wrapper function 
******************************************************/
int shmem_int_cswap_jw_( int * addr, int match, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap_jw wrapper function 
******************************************************/
int shmem_int_cswap_jw__( int * addr, int match, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap_jw( addr, match, value, pe) ; 
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


/******************************************************
***      shmem_long_cswap_jw wrapper function 
******************************************************/
long shmem_long_cswap_jw( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap_jw wrapper function 
******************************************************/
long SHMEM_LONG_CSWAP_JW( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap_jw wrapper function 
******************************************************/
long shmem_long_cswap_jw_( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap_jw wrapper function 
******************************************************/
long shmem_long_cswap_jw__( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_cswap wrapper function 
******************************************************/
long long shmem_longlong_cswap( long long * target, long long cond, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap( target, cond, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap( target, cond, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap wrapper function 
******************************************************/
long long SHMEM_LONGLONG_CSWAP( long long * target, long long cond, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap( target, cond, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap( target, cond, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap wrapper function 
******************************************************/
long long shmem_longlong_cswap_( long long * target, long long cond, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap( target, cond, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap( target, cond, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap wrapper function 
******************************************************/
long long shmem_longlong_cswap__( long long * target, long long cond, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap( target, cond, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap( target, cond, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_cswap_jw wrapper function 
******************************************************/
long long shmem_longlong_cswap_jw( long long * target, long long cond, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap_jw( target, cond, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap_jw( target, cond, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap_jw wrapper function 
******************************************************/
long long SHMEM_LONGLONG_CSWAP_JW( long long * target, long long cond, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap_jw( target, cond, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap_jw( target, cond, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap_jw wrapper function 
******************************************************/
long long shmem_longlong_cswap_jw_( long long * target, long long cond, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap_jw( target, cond, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap_jw( target, cond, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap_jw wrapper function 
******************************************************/
long long shmem_longlong_cswap_jw__( long long * target, long long cond, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap_jw( target, cond, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap_jw( target, cond, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_finc wrapper function 
******************************************************/
char shmem_char_finc( char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_finc wrapper function 
******************************************************/
char SHMEM_CHAR_FINC( char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_finc wrapper function 
******************************************************/
char shmem_char_finc_( char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_finc wrapper function 
******************************************************/
char shmem_char_finc__( char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_finc_jw wrapper function 
******************************************************/
char shmem_char_finc_jw( char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_finc_jw wrapper function 
******************************************************/
char SHMEM_CHAR_FINC_JW( char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_finc_jw wrapper function 
******************************************************/
char shmem_char_finc_jw_( char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_finc_jw wrapper function 
******************************************************/
char shmem_char_finc_jw__( char * addr, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_finc wrapper function 
******************************************************/
short shmem_short_finc( short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc wrapper function 
******************************************************/
short SHMEM_SHORT_FINC( short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc wrapper function 
******************************************************/
short shmem_short_finc_( short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc wrapper function 
******************************************************/
short shmem_short_finc__( short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_finc_jw wrapper function 
******************************************************/
short shmem_short_finc_jw( short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc_jw wrapper function 
******************************************************/
short SHMEM_SHORT_FINC_JW( short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc_jw wrapper function 
******************************************************/
short shmem_short_finc_jw_( short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc_jw wrapper function 
******************************************************/
short shmem_short_finc_jw__( short * addr, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


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
***      shmem_int_finc_jw wrapper function 
******************************************************/
int shmem_int_finc_jw( int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc_jw wrapper function 
******************************************************/
int SHMEM_INT_FINC_JW( int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc_jw wrapper function 
******************************************************/
int shmem_int_finc_jw_( int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc_jw wrapper function 
******************************************************/
int shmem_int_finc_jw__( int * addr, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc_jw( addr, pe) ; 
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
***      shmem_long_finc_jw wrapper function 
******************************************************/
long shmem_long_finc_jw( long * target, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_finc_jw( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_finc_jw( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_finc_jw wrapper function 
******************************************************/
long SHMEM_LONG_FINC_JW( long * target, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_finc_jw( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_finc_jw( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_finc_jw wrapper function 
******************************************************/
long shmem_long_finc_jw_( long * target, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_finc_jw( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_finc_jw( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_finc_jw wrapper function 
******************************************************/
long shmem_long_finc_jw__( long * target, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_finc_jw( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_finc_jw( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_finc wrapper function 
******************************************************/
long long shmem_longlong_finc( long long * target, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_finc wrapper function 
******************************************************/
long long SHMEM_LONGLONG_FINC( long long * target, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_finc wrapper function 
******************************************************/
long long shmem_longlong_finc_( long long * target, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_finc wrapper function 
******************************************************/
long long shmem_longlong_finc__( long long * target, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_finc_jw wrapper function 
******************************************************/
long long shmem_longlong_finc_jw( long long * target, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_finc_jw( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_finc_jw( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_finc_jw wrapper function 
******************************************************/
long long SHMEM_LONGLONG_FINC_JW( long long * target, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_finc_jw( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_finc_jw( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_finc_jw wrapper function 
******************************************************/
long long shmem_longlong_finc_jw_( long long * target, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_finc_jw( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_finc_jw( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_finc_jw wrapper function 
******************************************************/
long long shmem_longlong_finc_jw__( long long * target, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_finc_jw( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_finc_jw( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_fadd wrapper function 
******************************************************/
char shmem_char_fadd( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_fadd wrapper function 
******************************************************/
char SHMEM_CHAR_FADD( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_fadd wrapper function 
******************************************************/
char shmem_char_fadd_( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_fadd wrapper function 
******************************************************/
char shmem_char_fadd__( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_fadd_jw wrapper function 
******************************************************/
char shmem_char_fadd_jw( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_fadd_jw wrapper function 
******************************************************/
char SHMEM_CHAR_FADD_JW( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_fadd_jw wrapper function 
******************************************************/
char shmem_char_fadd_jw_( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_char_fadd_jw wrapper function 
******************************************************/
char shmem_char_fadd_jw__( char * addr, char value, int pe)
{
  char retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_char_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_char_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_char_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_fadd wrapper function 
******************************************************/
short shmem_short_fadd( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd wrapper function 
******************************************************/
short SHMEM_SHORT_FADD( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd wrapper function 
******************************************************/
short shmem_short_fadd_( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd wrapper function 
******************************************************/
short shmem_short_fadd__( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_fadd_jw wrapper function 
******************************************************/
short shmem_short_fadd_jw( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd_jw wrapper function 
******************************************************/
short SHMEM_SHORT_FADD_JW( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd_jw wrapper function 
******************************************************/
short shmem_short_fadd_jw_( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd_jw wrapper function 
******************************************************/
short shmem_short_fadd_jw__( short * addr, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd_jw( addr, value, pe) ; 
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
***      shmem_int_fadd_jw wrapper function 
******************************************************/
int shmem_int_fadd_jw( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd_jw wrapper function 
******************************************************/
int SHMEM_INT_FADD_JW( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd_jw wrapper function 
******************************************************/
int shmem_int_fadd_jw_( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd_jw wrapper function 
******************************************************/
int shmem_int_fadd_jw__( int * addr, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd_jw( addr, value, pe) ; 
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
***      shmem_long_fadd_jw wrapper function 
******************************************************/
long shmem_long_fadd_jw( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_fadd_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_fadd_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_fadd_jw wrapper function 
******************************************************/
long SHMEM_LONG_FADD_JW( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_fadd_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_fadd_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_fadd_jw wrapper function 
******************************************************/
long shmem_long_fadd_jw_( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_fadd_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_fadd_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_fadd_jw wrapper function 
******************************************************/
long shmem_long_fadd_jw__( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_fadd_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_fadd_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_fadd wrapper function 
******************************************************/
long long shmem_longlong_fadd( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_fadd wrapper function 
******************************************************/
long long SHMEM_LONGLONG_FADD( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_fadd wrapper function 
******************************************************/
long long shmem_longlong_fadd_( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_fadd wrapper function 
******************************************************/
long long shmem_longlong_fadd__( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_fadd_jw wrapper function 
******************************************************/
long long shmem_longlong_fadd_jw( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_fadd_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_fadd_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_fadd_jw wrapper function 
******************************************************/
long long SHMEM_LONGLONG_FADD_JW( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_fadd_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_fadd_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_fadd_jw wrapper function 
******************************************************/
long long shmem_longlong_fadd_jw_( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_fadd_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_fadd_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_fadd_jw wrapper function 
******************************************************/
long long shmem_longlong_fadd_jw__( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_fadd_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_fadd_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_inc wrapper function 
******************************************************/
void shmem_char_inc( char * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_inc wrapper function 
******************************************************/
void SHMEM_CHAR_INC( char * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_inc wrapper function 
******************************************************/
void shmem_char_inc_( char * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_inc wrapper function 
******************************************************/
void shmem_char_inc__( char * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_inc_jw wrapper function 
******************************************************/
void shmem_char_inc_jw( char * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_inc_jw wrapper function 
******************************************************/
void SHMEM_CHAR_INC_JW( char * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_inc_jw wrapper function 
******************************************************/
void shmem_char_inc_jw_( char * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_inc_jw wrapper function 
******************************************************/
void shmem_char_inc_jw__( char * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_inc wrapper function 
******************************************************/
void shmem_short_inc( short * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc wrapper function 
******************************************************/
void SHMEM_SHORT_INC( short * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc wrapper function 
******************************************************/
void shmem_short_inc_( short * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc wrapper function 
******************************************************/
void shmem_short_inc__( short * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_inc_jw wrapper function 
******************************************************/
void shmem_short_inc_jw( short * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc_jw wrapper function 
******************************************************/
void SHMEM_SHORT_INC_JW( short * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc_jw wrapper function 
******************************************************/
void shmem_short_inc_jw_( short * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc_jw wrapper function 
******************************************************/
void shmem_short_inc_jw__( short * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_inc wrapper function 
******************************************************/
void shmem_int_inc( int * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc wrapper function 
******************************************************/
void SHMEM_INT_INC( int * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc wrapper function 
******************************************************/
void shmem_int_inc_( int * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc wrapper function 
******************************************************/
void shmem_int_inc__( int * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_inc_jw wrapper function 
******************************************************/
void shmem_int_inc_jw( int * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc_jw wrapper function 
******************************************************/
void SHMEM_INT_INC_JW( int * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc_jw wrapper function 
******************************************************/
void shmem_int_inc_jw_( int * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc_jw wrapper function 
******************************************************/
void shmem_int_inc_jw__( int * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_inc wrapper function 
******************************************************/
void shmem_long_inc( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_inc wrapper function 
******************************************************/
void SHMEM_LONG_INC( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_inc wrapper function 
******************************************************/
void shmem_long_inc_( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_inc wrapper function 
******************************************************/
void shmem_long_inc__( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_inc_jw wrapper function 
******************************************************/
void shmem_long_inc_jw( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_inc_jw wrapper function 
******************************************************/
void SHMEM_LONG_INC_JW( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_inc_jw wrapper function 
******************************************************/
void shmem_long_inc_jw_( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_inc_jw wrapper function 
******************************************************/
void shmem_long_inc_jw__( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_inc wrapper function 
******************************************************/
void shmem_longlong_inc( long long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_inc wrapper function 
******************************************************/
void SHMEM_LONGLONG_INC( long long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_inc wrapper function 
******************************************************/
void shmem_longlong_inc_( long long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_inc wrapper function 
******************************************************/
void shmem_longlong_inc__( long long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_inc_jw wrapper function 
******************************************************/
void shmem_longlong_inc_jw( long long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_inc_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_INC_JW( long long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_inc_jw wrapper function 
******************************************************/
void shmem_longlong_inc_jw_( long long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_inc_jw wrapper function 
******************************************************/
void shmem_longlong_inc_jw__( long long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_add wrapper function 
******************************************************/
void shmem_char_add( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_add wrapper function 
******************************************************/
void SHMEM_CHAR_ADD( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_add wrapper function 
******************************************************/
void shmem_char_add_( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_add wrapper function 
******************************************************/
void shmem_char_add__( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_char_add_jw wrapper function 
******************************************************/
void shmem_char_add_jw( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_add_jw wrapper function 
******************************************************/
void SHMEM_CHAR_ADD_JW( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_add_jw wrapper function 
******************************************************/
void shmem_char_add_jw_( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_char_add_jw wrapper function 
******************************************************/
void shmem_char_add_jw__( char * addr, char value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_char_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_char_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_char_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_add wrapper function 
******************************************************/
void shmem_short_add( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_add wrapper function 
******************************************************/
void SHMEM_SHORT_ADD( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_add wrapper function 
******************************************************/
void shmem_short_add_( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_add wrapper function 
******************************************************/
void shmem_short_add__( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_add_jw wrapper function 
******************************************************/
void shmem_short_add_jw( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_add_jw wrapper function 
******************************************************/
void SHMEM_SHORT_ADD_JW( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_add_jw wrapper function 
******************************************************/
void shmem_short_add_jw_( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_add_jw wrapper function 
******************************************************/
void shmem_short_add_jw__( short * addr, short value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_add wrapper function 
******************************************************/
void shmem_int_add( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add wrapper function 
******************************************************/
void SHMEM_INT_ADD( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add wrapper function 
******************************************************/
void shmem_int_add_( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add wrapper function 
******************************************************/
void shmem_int_add__( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_add_jw wrapper function 
******************************************************/
void shmem_int_add_jw( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add_jw wrapper function 
******************************************************/
void SHMEM_INT_ADD_JW( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add_jw wrapper function 
******************************************************/
void shmem_int_add_jw_( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add_jw wrapper function 
******************************************************/
void shmem_int_add_jw__( int * addr, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_add wrapper function 
******************************************************/
void shmem_long_add( long * target, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_add wrapper function 
******************************************************/
void SHMEM_LONG_ADD( long * target, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_add wrapper function 
******************************************************/
void shmem_long_add_( long * target, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_add wrapper function 
******************************************************/
void shmem_long_add__( long * target, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_add_jw wrapper function 
******************************************************/
void shmem_long_add_jw( long * target, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_add_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_add_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_add_jw wrapper function 
******************************************************/
void SHMEM_LONG_ADD_JW( long * target, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_add_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_add_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_add_jw wrapper function 
******************************************************/
void shmem_long_add_jw_( long * target, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_add_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_add_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_add_jw wrapper function 
******************************************************/
void shmem_long_add_jw__( long * target, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_add_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_add_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_add wrapper function 
******************************************************/
void shmem_longlong_add( long long * target, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_add wrapper function 
******************************************************/
void SHMEM_LONGLONG_ADD( long long * target, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_add wrapper function 
******************************************************/
void shmem_longlong_add_( long long * target, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_add wrapper function 
******************************************************/
void shmem_longlong_add__( long long * target, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_add_jw wrapper function 
******************************************************/
void shmem_longlong_add_jw( long long * target, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_add_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_add_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_add_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_ADD_JW( long long * target, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_add_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_add_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_add_jw wrapper function 
******************************************************/
void shmem_longlong_add_jw_( long long * target, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_add_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_add_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_add_jw wrapper function 
******************************************************/
void shmem_longlong_add_jw__( long long * target, long long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_add_jw( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_add_jw( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
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
***      shmem_barrier_all_jw wrapper function 
******************************************************/
void shmem_barrier_all_jw( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_all_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_all_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_all_jw wrapper function 
******************************************************/
void SHMEM_BARRIER_ALL_JW( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_all_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_all_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_all_jw wrapper function 
******************************************************/
void shmem_barrier_all_jw_( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_all_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_all_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_all_jw wrapper function 
******************************************************/
void shmem_barrier_all_jw__( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_all_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_all_jw( ) ; 
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
***      shmem_barrier_jw wrapper function 
******************************************************/
void shmem_barrier_jw( int pestart, int log_pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_jw( pestart, log_pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_jw( pestart, log_pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_jw wrapper function 
******************************************************/
void SHMEM_BARRIER_JW( int pestart, int log_pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_jw( pestart, log_pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_jw( pestart, log_pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_jw wrapper function 
******************************************************/
void shmem_barrier_jw_( int pestart, int log_pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_jw( pestart, log_pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_jw( pestart, log_pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_jw wrapper function 
******************************************************/
void shmem_barrier_jw__( int pestart, int log_pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_jw( pestart, log_pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_jw( pestart, log_pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_barrier_ps wrapper function 
******************************************************/
void shmem_barrier_ps( int pestart, int pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_ps()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_ps( pestart, pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_ps( pestart, pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_ps wrapper function 
******************************************************/
void SHMEM_BARRIER_PS( int pestart, int pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_ps()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_ps( pestart, pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_ps( pestart, pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_ps wrapper function 
******************************************************/
void shmem_barrier_ps_( int pestart, int pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_ps()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_ps( pestart, pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_ps( pestart, pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_ps wrapper function 
******************************************************/
void shmem_barrier_ps__( int pestart, int pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_ps()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_ps( pestart, pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_ps( pestart, pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_barrier_ps_jw wrapper function 
******************************************************/
void shmem_barrier_ps_jw( int pestart, int pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_ps_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_ps_jw( pestart, pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_ps_jw( pestart, pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_ps_jw wrapper function 
******************************************************/
void SHMEM_BARRIER_PS_JW( int pestart, int pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_ps_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_ps_jw( pestart, pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_ps_jw( pestart, pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_ps_jw wrapper function 
******************************************************/
void shmem_barrier_ps_jw_( int pestart, int pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_ps_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_ps_jw( pestart, pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_ps_jw( pestart, pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_ps_jw wrapper function 
******************************************************/
void shmem_barrier_ps_jw__( int pestart, int pestride, int pesize, long * bar)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_ps_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_ps_jw( pestart, pestride, pesize, bar) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_ps_jw( pestart, pestride, pesize, bar) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_barrier_notify wrapper function 
******************************************************/
void shmem_barrier_notify( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_notify()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_notify( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_notify( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_notify wrapper function 
******************************************************/
void SHMEM_BARRIER_NOTIFY( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_notify()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_notify( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_notify( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_notify wrapper function 
******************************************************/
void shmem_barrier_notify_( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_notify()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_notify( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_notify( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_notify wrapper function 
******************************************************/
void shmem_barrier_notify__( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_notify()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_notify( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_notify( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_barrier_notify_jw wrapper function 
******************************************************/
void shmem_barrier_notify_jw( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_notify_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_notify_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_notify_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_notify_jw wrapper function 
******************************************************/
void SHMEM_BARRIER_NOTIFY_JW( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_notify_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_notify_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_notify_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_notify_jw wrapper function 
******************************************************/
void shmem_barrier_notify_jw_( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_notify_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_notify_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_notify_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_notify_jw wrapper function 
******************************************************/
void shmem_barrier_notify_jw__( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_notify_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_notify_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_notify_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_barrier_wait wrapper function 
******************************************************/
void shmem_barrier_wait( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_wait( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_wait( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_wait wrapper function 
******************************************************/
void SHMEM_BARRIER_WAIT( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_wait( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_wait( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_wait wrapper function 
******************************************************/
void shmem_barrier_wait_( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_wait( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_wait( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_wait wrapper function 
******************************************************/
void shmem_barrier_wait__( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_wait( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_wait( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_barrier_wait_jw wrapper function 
******************************************************/
void shmem_barrier_wait_jw( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_wait_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_wait_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_wait_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_wait_jw wrapper function 
******************************************************/
void SHMEM_BARRIER_WAIT_JW( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_wait_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_wait_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_wait_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_wait_jw wrapper function 
******************************************************/
void shmem_barrier_wait_jw_( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_wait_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_wait_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_wait_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier_wait_jw wrapper function 
******************************************************/
void shmem_barrier_wait_jw__( )
{

  TAU_PROFILE_TIMER(t, "shmem_barrier_wait_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier_wait_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier_wait_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


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
***      shmem_set_lock_jw wrapper function 
******************************************************/
void shmem_set_lock_jw( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_set_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_lock_jw wrapper function 
******************************************************/
void SHMEM_SET_LOCK_JW( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_set_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_lock_jw wrapper function 
******************************************************/
void shmem_set_lock_jw_( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_set_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_lock_jw wrapper function 
******************************************************/
void shmem_set_lock_jw__( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_set_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_lock_jw( lock) ; 
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
***      shmem_clear_lock_jw wrapper function 
******************************************************/
void shmem_clear_lock_jw( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_lock_jw wrapper function 
******************************************************/
void SHMEM_CLEAR_LOCK_JW( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_lock_jw wrapper function 
******************************************************/
void shmem_clear_lock_jw_( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_lock_jw wrapper function 
******************************************************/
void shmem_clear_lock_jw__( long * lock)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_lock_jw( lock) ; 
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


/******************************************************
***      shmem_test_lock_jw wrapper function 
******************************************************/
int shmem_test_lock_jw( long * lock)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_lock_jw wrapper function 
******************************************************/
int SHMEM_TEST_LOCK_JW( long * lock)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_lock_jw wrapper function 
******************************************************/
int shmem_test_lock_jw_( long * lock)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_test_lock_jw wrapper function 
******************************************************/
int shmem_test_lock_jw__( long * lock)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_test_lock_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_test_lock_jw( lock) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_test_lock_jw( lock) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


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
***      shmem_short_sum_to_all_jw wrapper function 
******************************************************/
void shmem_short_sum_to_all_jw( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_SHORT_SUM_TO_ALL_JW( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all_jw wrapper function 
******************************************************/
void shmem_short_sum_to_all_jw_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all_jw wrapper function 
******************************************************/
void shmem_short_sum_to_all_jw__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_short_max_to_all_jw wrapper function 
******************************************************/
void shmem_short_max_to_all_jw( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all_jw wrapper function 
******************************************************/
void SHMEM_SHORT_MAX_TO_ALL_JW( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all_jw wrapper function 
******************************************************/
void shmem_short_max_to_all_jw_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all_jw wrapper function 
******************************************************/
void shmem_short_max_to_all_jw__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_short_min_to_all_jw wrapper function 
******************************************************/
void shmem_short_min_to_all_jw( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all_jw wrapper function 
******************************************************/
void SHMEM_SHORT_MIN_TO_ALL_JW( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all_jw wrapper function 
******************************************************/
void shmem_short_min_to_all_jw_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all_jw wrapper function 
******************************************************/
void shmem_short_min_to_all_jw__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_short_prod_to_all_jw wrapper function 
******************************************************/
void shmem_short_prod_to_all_jw( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_SHORT_PROD_TO_ALL_JW( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all_jw wrapper function 
******************************************************/
void shmem_short_prod_to_all_jw_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all_jw wrapper function 
******************************************************/
void shmem_short_prod_to_all_jw__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_short_and_to_all_jw wrapper function 
******************************************************/
void shmem_short_and_to_all_jw( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all_jw wrapper function 
******************************************************/
void SHMEM_SHORT_AND_TO_ALL_JW( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all_jw wrapper function 
******************************************************/
void shmem_short_and_to_all_jw_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all_jw wrapper function 
******************************************************/
void shmem_short_and_to_all_jw__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_short_or_to_all_jw wrapper function 
******************************************************/
void shmem_short_or_to_all_jw( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all_jw wrapper function 
******************************************************/
void SHMEM_SHORT_OR_TO_ALL_JW( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all_jw wrapper function 
******************************************************/
void shmem_short_or_to_all_jw_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all_jw wrapper function 
******************************************************/
void shmem_short_or_to_all_jw__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_short_xor_to_all_jw wrapper function 
******************************************************/
void shmem_short_xor_to_all_jw( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all_jw wrapper function 
******************************************************/
void SHMEM_SHORT_XOR_TO_ALL_JW( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all_jw wrapper function 
******************************************************/
void shmem_short_xor_to_all_jw_( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all_jw wrapper function 
******************************************************/
void shmem_short_xor_to_all_jw__( short * trg, const short * src, size_t nreduce, int pestart, int log_pestride, int pesize, short * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_int_sum_to_all_jw wrapper function 
******************************************************/
void shmem_int_sum_to_all_jw( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_INT_SUM_TO_ALL_JW( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all_jw wrapper function 
******************************************************/
void shmem_int_sum_to_all_jw_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all_jw wrapper function 
******************************************************/
void shmem_int_sum_to_all_jw__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_int_max_to_all_jw wrapper function 
******************************************************/
void shmem_int_max_to_all_jw( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all_jw wrapper function 
******************************************************/
void SHMEM_INT_MAX_TO_ALL_JW( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all_jw wrapper function 
******************************************************/
void shmem_int_max_to_all_jw_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all_jw wrapper function 
******************************************************/
void shmem_int_max_to_all_jw__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_int_min_to_all_jw wrapper function 
******************************************************/
void shmem_int_min_to_all_jw( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all_jw wrapper function 
******************************************************/
void SHMEM_INT_MIN_TO_ALL_JW( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all_jw wrapper function 
******************************************************/
void shmem_int_min_to_all_jw_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all_jw wrapper function 
******************************************************/
void shmem_int_min_to_all_jw__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_int_prod_to_all_jw wrapper function 
******************************************************/
void shmem_int_prod_to_all_jw( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_INT_PROD_TO_ALL_JW( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all_jw wrapper function 
******************************************************/
void shmem_int_prod_to_all_jw_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all_jw wrapper function 
******************************************************/
void shmem_int_prod_to_all_jw__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_int_and_to_all_jw wrapper function 
******************************************************/
void shmem_int_and_to_all_jw( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all_jw wrapper function 
******************************************************/
void SHMEM_INT_AND_TO_ALL_JW( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all_jw wrapper function 
******************************************************/
void shmem_int_and_to_all_jw_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all_jw wrapper function 
******************************************************/
void shmem_int_and_to_all_jw__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_int_or_to_all_jw wrapper function 
******************************************************/
void shmem_int_or_to_all_jw( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all_jw wrapper function 
******************************************************/
void SHMEM_INT_OR_TO_ALL_JW( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all_jw wrapper function 
******************************************************/
void shmem_int_or_to_all_jw_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all_jw wrapper function 
******************************************************/
void shmem_int_or_to_all_jw__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_int_xor_to_all_jw wrapper function 
******************************************************/
void shmem_int_xor_to_all_jw( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all_jw wrapper function 
******************************************************/
void SHMEM_INT_XOR_TO_ALL_JW( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all_jw wrapper function 
******************************************************/
void shmem_int_xor_to_all_jw_( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all_jw wrapper function 
******************************************************/
void shmem_int_xor_to_all_jw__( int * trg, const int * src, size_t nreduce, int pestart, int log_pestride, int pesize, int * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_long_sum_to_all_jw wrapper function 
******************************************************/
void shmem_long_sum_to_all_jw( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONG_SUM_TO_ALL_JW( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all_jw wrapper function 
******************************************************/
void shmem_long_sum_to_all_jw_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all_jw wrapper function 
******************************************************/
void shmem_long_sum_to_all_jw__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_long_max_to_all_jw wrapper function 
******************************************************/
void shmem_long_max_to_all_jw( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONG_MAX_TO_ALL_JW( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all_jw wrapper function 
******************************************************/
void shmem_long_max_to_all_jw_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all_jw wrapper function 
******************************************************/
void shmem_long_max_to_all_jw__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_long_min_to_all_jw wrapper function 
******************************************************/
void shmem_long_min_to_all_jw( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONG_MIN_TO_ALL_JW( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all_jw wrapper function 
******************************************************/
void shmem_long_min_to_all_jw_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all_jw wrapper function 
******************************************************/
void shmem_long_min_to_all_jw__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_long_prod_to_all_jw wrapper function 
******************************************************/
void shmem_long_prod_to_all_jw( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONG_PROD_TO_ALL_JW( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all_jw wrapper function 
******************************************************/
void shmem_long_prod_to_all_jw_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all_jw wrapper function 
******************************************************/
void shmem_long_prod_to_all_jw__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_long_and_to_all_jw wrapper function 
******************************************************/
void shmem_long_and_to_all_jw( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONG_AND_TO_ALL_JW( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all_jw wrapper function 
******************************************************/
void shmem_long_and_to_all_jw_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all_jw wrapper function 
******************************************************/
void shmem_long_and_to_all_jw__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_long_or_to_all_jw wrapper function 
******************************************************/
void shmem_long_or_to_all_jw( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONG_OR_TO_ALL_JW( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all_jw wrapper function 
******************************************************/
void shmem_long_or_to_all_jw_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all_jw wrapper function 
******************************************************/
void shmem_long_or_to_all_jw__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_long_xor_to_all_jw wrapper function 
******************************************************/
void shmem_long_xor_to_all_jw( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONG_XOR_TO_ALL_JW( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all_jw wrapper function 
******************************************************/
void shmem_long_xor_to_all_jw_( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all_jw wrapper function 
******************************************************/
void shmem_long_xor_to_all_jw__( long * trg, const long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_longlong_sum_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_sum_to_all_jw( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_SUM_TO_ALL_JW( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_sum_to_all_jw_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_sum_to_all_jw__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_longlong_max_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_max_to_all_jw( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_MAX_TO_ALL_JW( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_max_to_all_jw_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_max_to_all_jw__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_longlong_min_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_min_to_all_jw( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_MIN_TO_ALL_JW( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_min_to_all_jw_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_min_to_all_jw__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_longlong_prod_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_prod_to_all_jw( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_PROD_TO_ALL_JW( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_prod_to_all_jw_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_prod_to_all_jw__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_longlong_and_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_and_to_all_jw( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_AND_TO_ALL_JW( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_and_to_all_jw_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_and_to_all_jw__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_longlong_or_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_or_to_all_jw( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_OR_TO_ALL_JW( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_or_to_all_jw_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_or_to_all_jw__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_longlong_xor_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_xor_to_all_jw( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGLONG_XOR_TO_ALL_JW( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_xor_to_all_jw_( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all_jw wrapper function 
******************************************************/
void shmem_longlong_xor_to_all_jw__( long long * trg, const long long * src, size_t nreduce, int pestart, int log_pestride, int pesize, long long * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_float_sum_to_all_jw wrapper function 
******************************************************/
void shmem_float_sum_to_all_jw( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_SUM_TO_ALL_JW( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all_jw wrapper function 
******************************************************/
void shmem_float_sum_to_all_jw_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all_jw wrapper function 
******************************************************/
void shmem_float_sum_to_all_jw__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_float_max_to_all_jw wrapper function 
******************************************************/
void shmem_float_max_to_all_jw( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_MAX_TO_ALL_JW( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all_jw wrapper function 
******************************************************/
void shmem_float_max_to_all_jw_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all_jw wrapper function 
******************************************************/
void shmem_float_max_to_all_jw__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_float_min_to_all_jw wrapper function 
******************************************************/
void shmem_float_min_to_all_jw( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_MIN_TO_ALL_JW( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all_jw wrapper function 
******************************************************/
void shmem_float_min_to_all_jw_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all_jw wrapper function 
******************************************************/
void shmem_float_min_to_all_jw__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_float_prod_to_all_jw wrapper function 
******************************************************/
void shmem_float_prod_to_all_jw( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_FLOAT_PROD_TO_ALL_JW( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all_jw wrapper function 
******************************************************/
void shmem_float_prod_to_all_jw_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all_jw wrapper function 
******************************************************/
void shmem_float_prod_to_all_jw__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_double_sum_to_all_jw wrapper function 
******************************************************/
void shmem_double_sum_to_all_jw( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_SUM_TO_ALL_JW( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all_jw wrapper function 
******************************************************/
void shmem_double_sum_to_all_jw_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all_jw wrapper function 
******************************************************/
void shmem_double_sum_to_all_jw__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_double_max_to_all_jw wrapper function 
******************************************************/
void shmem_double_max_to_all_jw( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_MAX_TO_ALL_JW( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all_jw wrapper function 
******************************************************/
void shmem_double_max_to_all_jw_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all_jw wrapper function 
******************************************************/
void shmem_double_max_to_all_jw__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_double_min_to_all_jw wrapper function 
******************************************************/
void shmem_double_min_to_all_jw( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_MIN_TO_ALL_JW( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all_jw wrapper function 
******************************************************/
void shmem_double_min_to_all_jw_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all_jw wrapper function 
******************************************************/
void shmem_double_min_to_all_jw__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
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
***      shmem_double_prod_to_all_jw wrapper function 
******************************************************/
void shmem_double_prod_to_all_jw( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_DOUBLE_PROD_TO_ALL_JW( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all_jw wrapper function 
******************************************************/
void shmem_double_prod_to_all_jw_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all_jw wrapper function 
******************************************************/
void shmem_double_prod_to_all_jw__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_sum_to_all wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_SUM_TO_ALL( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all_( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all__( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_sum_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all_jw( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_SUM_TO_ALL_JW( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all_jw_( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all_jw__( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_max_to_all wrapper function 
******************************************************/
void shmem_longdouble_max_to_all( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_MAX_TO_ALL( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all wrapper function 
******************************************************/
void shmem_longdouble_max_to_all_( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all wrapper function 
******************************************************/
void shmem_longdouble_max_to_all__( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_max_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_max_to_all_jw( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_MAX_TO_ALL_JW( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_max_to_all_jw_( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_max_to_all_jw__( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_min_to_all wrapper function 
******************************************************/
void shmem_longdouble_min_to_all( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_MIN_TO_ALL( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all wrapper function 
******************************************************/
void shmem_longdouble_min_to_all_( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all wrapper function 
******************************************************/
void shmem_longdouble_min_to_all__( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_min_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_min_to_all_jw( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_MIN_TO_ALL_JW( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_min_to_all_jw_( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_min_to_all_jw__( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_prod_to_all wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_PROD_TO_ALL( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all_( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all__( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_prod_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all_jw( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_PROD_TO_ALL_JW( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all_jw_( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all_jw wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all_jw__( long double * trg, const long double * src, size_t nreduce, int pestart, int log_pestride, int pesize, long double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_sum_to_all wrapper function 
******************************************************/
void shmem_complexf_sum_to_all( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXF_SUM_TO_ALL( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all wrapper function 
******************************************************/
void shmem_complexf_sum_to_all_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all wrapper function 
******************************************************/
void shmem_complexf_sum_to_all__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_sum_to_all_jw wrapper function 
******************************************************/
void shmem_complexf_sum_to_all_jw( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_COMPLEXF_SUM_TO_ALL_JW( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all_jw wrapper function 
******************************************************/
void shmem_complexf_sum_to_all_jw_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all_jw wrapper function 
******************************************************/
void shmem_complexf_sum_to_all_jw__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_prod_to_all wrapper function 
******************************************************/
void shmem_complexf_prod_to_all( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXF_PROD_TO_ALL( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all wrapper function 
******************************************************/
void shmem_complexf_prod_to_all_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all wrapper function 
******************************************************/
void shmem_complexf_prod_to_all__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_prod_to_all_jw wrapper function 
******************************************************/
void shmem_complexf_prod_to_all_jw( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_COMPLEXF_PROD_TO_ALL_JW( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all_jw wrapper function 
******************************************************/
void shmem_complexf_prod_to_all_jw_( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all_jw wrapper function 
******************************************************/
void shmem_complexf_prod_to_all_jw__( float * trg, const float * src, size_t nreduce, int pestart, int log_pestride, int pesize, float * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_sum_to_all wrapper function 
******************************************************/
void shmem_complexd_sum_to_all( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXD_SUM_TO_ALL( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all wrapper function 
******************************************************/
void shmem_complexd_sum_to_all_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all wrapper function 
******************************************************/
void shmem_complexd_sum_to_all__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_sum_to_all_jw wrapper function 
******************************************************/
void shmem_complexd_sum_to_all_jw( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all_jw wrapper function 
******************************************************/
void SHMEM_COMPLEXD_SUM_TO_ALL_JW( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all_jw wrapper function 
******************************************************/
void shmem_complexd_sum_to_all_jw_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all_jw wrapper function 
******************************************************/
void shmem_complexd_sum_to_all_jw__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_prod_to_all wrapper function 
******************************************************/
void shmem_complexd_prod_to_all( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXD_PROD_TO_ALL( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all wrapper function 
******************************************************/
void shmem_complexd_prod_to_all_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all wrapper function 
******************************************************/
void shmem_complexd_prod_to_all__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_prod_to_all_jw wrapper function 
******************************************************/
void shmem_complexd_prod_to_all_jw( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all_jw wrapper function 
******************************************************/
void SHMEM_COMPLEXD_PROD_TO_ALL_JW( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all_jw wrapper function 
******************************************************/
void shmem_complexd_prod_to_all_jw_( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all_jw wrapper function 
******************************************************/
void shmem_complexd_prod_to_all_jw__( double * trg, const double * src, size_t nreduce, int pestart, int log_pestride, int pesize, double * pwrk, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all_jw( trg, src, nreduce, pestart, log_pestride, pesize, pwrk, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect32 wrapper function 
******************************************************/
void shmem_collect32( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32 wrapper function 
******************************************************/
void SHMEM_COLLECT32( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32 wrapper function 
******************************************************/
void shmem_collect32_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32 wrapper function 
******************************************************/
void shmem_collect32__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect32_jw wrapper function 
******************************************************/
void shmem_collect32_jw( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32_jw wrapper function 
******************************************************/
void SHMEM_COLLECT32_JW( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32_jw wrapper function 
******************************************************/
void shmem_collect32_jw_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32_jw wrapper function 
******************************************************/
void shmem_collect32_jw__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect64 wrapper function 
******************************************************/
void shmem_collect64( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64 wrapper function 
******************************************************/
void SHMEM_COLLECT64( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64 wrapper function 
******************************************************/
void shmem_collect64_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64 wrapper function 
******************************************************/
void shmem_collect64__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect64_jw wrapper function 
******************************************************/
void shmem_collect64_jw( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64_jw wrapper function 
******************************************************/
void SHMEM_COLLECT64_JW( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64_jw wrapper function 
******************************************************/
void shmem_collect64_jw_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64_jw wrapper function 
******************************************************/
void shmem_collect64_jw__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect32 wrapper function 
******************************************************/
void shmem_fcollect32( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32 wrapper function 
******************************************************/
void SHMEM_FCOLLECT32( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32 wrapper function 
******************************************************/
void shmem_fcollect32_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32 wrapper function 
******************************************************/
void shmem_fcollect32__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect32_jw wrapper function 
******************************************************/
void shmem_fcollect32_jw( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32_jw wrapper function 
******************************************************/
void SHMEM_FCOLLECT32_JW( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32_jw wrapper function 
******************************************************/
void shmem_fcollect32_jw_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32_jw wrapper function 
******************************************************/
void shmem_fcollect32_jw__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect64 wrapper function 
******************************************************/
void shmem_fcollect64( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64 wrapper function 
******************************************************/
void SHMEM_FCOLLECT64( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64 wrapper function 
******************************************************/
void shmem_fcollect64_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64 wrapper function 
******************************************************/
void shmem_fcollect64__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect64_jw wrapper function 
******************************************************/
void shmem_fcollect64_jw( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64_jw wrapper function 
******************************************************/
void SHMEM_FCOLLECT64_JW( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64_jw wrapper function 
******************************************************/
void shmem_fcollect64_jw_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64_jw wrapper function 
******************************************************/
void shmem_fcollect64_jw__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
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
***      shmem_broadcast32_jw wrapper function 
******************************************************/
void shmem_broadcast32_jw( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32_jw wrapper function 
******************************************************/
void SHMEM_BROADCAST32_JW( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32_jw wrapper function 
******************************************************/
void shmem_broadcast32_jw_( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32_jw wrapper function 
******************************************************/
void shmem_broadcast32_jw__( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
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
***      shmem_broadcast64_jw wrapper function 
******************************************************/
void shmem_broadcast64_jw( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64_jw wrapper function 
******************************************************/
void SHMEM_BROADCAST64_JW( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64_jw wrapper function 
******************************************************/
void shmem_broadcast64_jw_( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64_jw wrapper function 
******************************************************/
void shmem_broadcast64_jw__( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
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
***      shmem_n_pes_jw wrapper function 
******************************************************/
int shmem_n_pes_jw( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_n_pes_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_n_pes_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_n_pes_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_n_pes_jw wrapper function 
******************************************************/
int SHMEM_N_PES_JW( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_n_pes_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_n_pes_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_n_pes_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_n_pes_jw wrapper function 
******************************************************/
int shmem_n_pes_jw_( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_n_pes_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_n_pes_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_n_pes_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_n_pes_jw wrapper function 
******************************************************/
int shmem_n_pes_jw__( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_n_pes_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_n_pes_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_n_pes_jw( ) ; 
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
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_my_pe_jw wrapper function 
******************************************************/
int shmem_my_pe_jw( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_my_pe_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_my_pe_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_my_pe_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_my_pe_jw wrapper function 
******************************************************/
int SHMEM_MY_PE_JW( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_my_pe_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_my_pe_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_my_pe_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_my_pe_jw wrapper function 
******************************************************/
int shmem_my_pe_jw_( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_my_pe_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_my_pe_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_my_pe_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_my_pe_jw wrapper function 
******************************************************/
int shmem_my_pe_jw__( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_my_pe_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_my_pe_jw( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_my_pe_jw( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_clear_cache_inv wrapper function 
******************************************************/
void shmem_clear_cache_inv( )
{

  TAU_PROFILE_TIMER(t, "shmem_clear_cache_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_cache_inv( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_cache_inv( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_cache_inv wrapper function 
******************************************************/
void SHMEM_CLEAR_CACHE_INV( )
{

  TAU_PROFILE_TIMER(t, "shmem_clear_cache_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_cache_inv( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_cache_inv( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_cache_inv wrapper function 
******************************************************/
void shmem_clear_cache_inv_( )
{

  TAU_PROFILE_TIMER(t, "shmem_clear_cache_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_cache_inv( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_cache_inv( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_cache_inv wrapper function 
******************************************************/
void shmem_clear_cache_inv__( )
{

  TAU_PROFILE_TIMER(t, "shmem_clear_cache_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_cache_inv( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_cache_inv( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_set_cache_inv wrapper function 
******************************************************/
void shmem_set_cache_inv( )
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_inv( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_inv( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_inv wrapper function 
******************************************************/
void SHMEM_SET_CACHE_INV( )
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_inv( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_inv( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_inv wrapper function 
******************************************************/
void shmem_set_cache_inv_( )
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_inv( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_inv( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_inv wrapper function 
******************************************************/
void shmem_set_cache_inv__( )
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_inv( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_inv( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_clear_cache_line_inv wrapper function 
******************************************************/
void shmem_clear_cache_line_inv( void * v)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_cache_line_inv( v) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_cache_line_inv( v) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_cache_line_inv wrapper function 
******************************************************/
void SHMEM_CLEAR_CACHE_LINE_INV( void * v)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_cache_line_inv( v) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_cache_line_inv( v) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_cache_line_inv wrapper function 
******************************************************/
void shmem_clear_cache_line_inv_( void * v)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_cache_line_inv( v) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_cache_line_inv( v) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_clear_cache_line_inv wrapper function 
******************************************************/
void shmem_clear_cache_line_inv__( void * v)
{

  TAU_PROFILE_TIMER(t, "shmem_clear_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_clear_cache_line_inv( v) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_clear_cache_line_inv( v) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_set_cache_line_inv wrapper function 
******************************************************/
void shmem_set_cache_line_inv( void * ptr)
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_line_inv( ptr) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_line_inv( ptr) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_line_inv wrapper function 
******************************************************/
void SHMEM_SET_CACHE_LINE_INV( void * ptr)
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_line_inv( ptr) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_line_inv( ptr) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_line_inv wrapper function 
******************************************************/
void shmem_set_cache_line_inv_( void * ptr)
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_line_inv( ptr) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_line_inv( ptr) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_line_inv wrapper function 
******************************************************/
void shmem_set_cache_line_inv__( void * ptr)
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_line_inv( ptr) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_line_inv( ptr) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_udcflush wrapper function 
******************************************************/
void shmem_udcflush( )
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush wrapper function 
******************************************************/
void SHMEM_UDCFLUSH( )
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush wrapper function 
******************************************************/
void shmem_udcflush_( )
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush wrapper function 
******************************************************/
void shmem_udcflush__( )
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_udcflush_line wrapper function 
******************************************************/
void shmem_udcflush_line( void * ptr)
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush_line()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush_line( ptr) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush_line( ptr) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush_line wrapper function 
******************************************************/
void SHMEM_UDCFLUSH_LINE( void * ptr)
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush_line()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush_line( ptr) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush_line( ptr) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush_line wrapper function 
******************************************************/
void shmem_udcflush_line_( void * ptr)
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush_line()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush_line( ptr) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush_line( ptr) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush_line wrapper function 
******************************************************/
void shmem_udcflush_line__( void * ptr)
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush_line()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush_line( ptr) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush_line( ptr) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
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


/******************************************************
***      shmem_ptr wrapper function 
******************************************************/
void * shmem_ptr( void * targ, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr( targ, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr( targ, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr wrapper function 
******************************************************/
void * SHMEM_PTR( void * targ, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr( targ, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr( targ, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr wrapper function 
******************************************************/
void * shmem_ptr_( void * targ, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr( targ, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr( targ, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr wrapper function 
******************************************************/
void * shmem_ptr__( void * targ, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr( targ, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr( targ, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ptr_jw wrapper function 
******************************************************/
void * shmem_ptr_jw( void * targ, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr_jw( targ, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr_jw( targ, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr_jw wrapper function 
******************************************************/
void * SHMEM_PTR_JW( void * targ, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr_jw( targ, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr_jw( targ, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr_jw wrapper function 
******************************************************/
void * shmem_ptr_jw_( void * targ, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr_jw( targ, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr_jw( targ, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr_jw wrapper function 
******************************************************/
void * shmem_ptr_jw__( void * targ, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr_jw( targ, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr_jw( targ, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_pe_accessible wrapper function 
******************************************************/
int shmem_pe_accessible( int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_pe_accessible()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_pe_accessible( pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_pe_accessible( pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_pe_accessible wrapper function 
******************************************************/
int SHMEM_PE_ACCESSIBLE( int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_pe_accessible()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_pe_accessible( pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_pe_accessible( pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_pe_accessible wrapper function 
******************************************************/
int shmem_pe_accessible_( int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_pe_accessible()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_pe_accessible( pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_pe_accessible( pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_pe_accessible wrapper function 
******************************************************/
int shmem_pe_accessible__( int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_pe_accessible()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_pe_accessible( pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_pe_accessible( pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_pe_accessible_jw wrapper function 
******************************************************/
int shmem_pe_accessible_jw( int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_pe_accessible_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_pe_accessible_jw( pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_pe_accessible_jw( pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_pe_accessible_jw wrapper function 
******************************************************/
int SHMEM_PE_ACCESSIBLE_JW( int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_pe_accessible_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_pe_accessible_jw( pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_pe_accessible_jw( pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_pe_accessible_jw wrapper function 
******************************************************/
int shmem_pe_accessible_jw_( int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_pe_accessible_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_pe_accessible_jw( pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_pe_accessible_jw( pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_pe_accessible_jw wrapper function 
******************************************************/
int shmem_pe_accessible_jw__( int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_pe_accessible_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_pe_accessible_jw( pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_pe_accessible_jw( pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_addr_accessible wrapper function 
******************************************************/
int shmem_addr_accessible( void * arg)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_addr_accessible()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_addr_accessible( arg) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_addr_accessible( arg) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_addr_accessible wrapper function 
******************************************************/
int SHMEM_ADDR_ACCESSIBLE( void * arg)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_addr_accessible()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_addr_accessible( arg) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_addr_accessible( arg) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_addr_accessible wrapper function 
******************************************************/
int shmem_addr_accessible_( void * arg)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_addr_accessible()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_addr_accessible( arg) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_addr_accessible( arg) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_addr_accessible wrapper function 
******************************************************/
int shmem_addr_accessible__( void * arg)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_addr_accessible()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_addr_accessible( arg) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_addr_accessible( arg) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_group_version wrapper function 
******************************************************/
int shmem_group_version( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_version( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_version( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_version wrapper function 
******************************************************/
int SHMEM_GROUP_VERSION( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_version( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_version( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_version wrapper function 
******************************************************/
int shmem_group_version_( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_version( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_version( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_version wrapper function 
******************************************************/
int shmem_group_version__( )
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_version( ) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_version( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_group_create_strided wrapper function 
******************************************************/
int shmem_group_create_strided( int PE_start, int PE_stride, int PE_size, int * racom, long * isync)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_create_strided()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_create_strided( PE_start, PE_stride, PE_size, racom, isync) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_create_strided( PE_start, PE_stride, PE_size, racom, isync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_create_strided wrapper function 
******************************************************/
int SHMEM_GROUP_CREATE_STRIDED( int PE_start, int PE_stride, int PE_size, int * racom, long * isync)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_create_strided()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_create_strided( PE_start, PE_stride, PE_size, racom, isync) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_create_strided( PE_start, PE_stride, PE_size, racom, isync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_create_strided wrapper function 
******************************************************/
int shmem_group_create_strided_( int PE_start, int PE_stride, int PE_size, int * racom, long * isync)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_create_strided()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_create_strided( PE_start, PE_stride, PE_size, racom, isync) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_create_strided( PE_start, PE_stride, PE_size, racom, isync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_create_strided wrapper function 
******************************************************/
int shmem_group_create_strided__( int PE_start, int PE_stride, int PE_size, int * racom, long * isync)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_create_strided()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_create_strided( PE_start, PE_stride, PE_size, racom, isync) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_create_strided( PE_start, PE_stride, PE_size, racom, isync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_group_create_strided_jw wrapper function 
******************************************************/
int shmem_group_create_strided_jw( int PE_start, int PE_stride, int PE_size, int * racom, long * isync)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_create_strided_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_create_strided_jw( PE_start, PE_stride, PE_size, racom, isync) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_create_strided_jw( PE_start, PE_stride, PE_size, racom, isync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_create_strided_jw wrapper function 
******************************************************/
int SHMEM_GROUP_CREATE_STRIDED_JW( int PE_start, int PE_stride, int PE_size, int * racom, long * isync)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_create_strided_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_create_strided_jw( PE_start, PE_stride, PE_size, racom, isync) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_create_strided_jw( PE_start, PE_stride, PE_size, racom, isync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_create_strided_jw wrapper function 
******************************************************/
int shmem_group_create_strided_jw_( int PE_start, int PE_stride, int PE_size, int * racom, long * isync)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_create_strided_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_create_strided_jw( PE_start, PE_stride, PE_size, racom, isync) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_create_strided_jw( PE_start, PE_stride, PE_size, racom, isync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_group_create_strided_jw wrapper function 
******************************************************/
int shmem_group_create_strided_jw__( int PE_start, int PE_stride, int PE_size, int * racom, long * isync)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_group_create_strided_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_group_create_strided_jw( PE_start, PE_stride, PE_size, racom, isync) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_group_create_strided_jw( PE_start, PE_stride, PE_size, racom, isync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_group_delete wrapper function 
******************************************************/
void shmem_group_delete( int handle)
{

  TAU_PROFILE_TIMER(t, "shmem_group_delete()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_delete( handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_delete( handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_delete wrapper function 
******************************************************/
void SHMEM_GROUP_DELETE( int handle)
{

  TAU_PROFILE_TIMER(t, "shmem_group_delete()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_delete( handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_delete( handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_delete wrapper function 
******************************************************/
void shmem_group_delete_( int handle)
{

  TAU_PROFILE_TIMER(t, "shmem_group_delete()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_delete( handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_delete( handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_delete wrapper function 
******************************************************/
void shmem_group_delete__( int handle)
{

  TAU_PROFILE_TIMER(t, "shmem_group_delete()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_delete( handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_delete( handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_group_delete_jw wrapper function 
******************************************************/
void shmem_group_delete_jw( int handle)
{

  TAU_PROFILE_TIMER(t, "shmem_group_delete_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_delete_jw( handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_delete_jw( handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_delete_jw wrapper function 
******************************************************/
void SHMEM_GROUP_DELETE_JW( int handle)
{

  TAU_PROFILE_TIMER(t, "shmem_group_delete_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_delete_jw( handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_delete_jw( handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_delete_jw wrapper function 
******************************************************/
void shmem_group_delete_jw_( int handle)
{

  TAU_PROFILE_TIMER(t, "shmem_group_delete_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_delete_jw( handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_delete_jw( handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_delete_jw wrapper function 
******************************************************/
void shmem_group_delete_jw__( int handle)
{

  TAU_PROFILE_TIMER(t, "shmem_group_delete_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_delete_jw( handle) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_delete_jw( handle) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_group_inquire wrapper function 
******************************************************/
void shmem_group_inquire( int handle, shmem_group_t * sgs)
{

  TAU_PROFILE_TIMER(t, "shmem_group_inquire()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_inquire( handle, sgs) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_inquire( handle, sgs) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_inquire wrapper function 
******************************************************/
void SHMEM_GROUP_INQUIRE( int handle, shmem_group_t * sgs)
{

  TAU_PROFILE_TIMER(t, "shmem_group_inquire()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_inquire( handle, sgs) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_inquire( handle, sgs) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_inquire wrapper function 
******************************************************/
void shmem_group_inquire_( int handle, shmem_group_t * sgs)
{

  TAU_PROFILE_TIMER(t, "shmem_group_inquire()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_inquire( handle, sgs) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_inquire( handle, sgs) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_inquire wrapper function 
******************************************************/
void shmem_group_inquire__( int handle, shmem_group_t * sgs)
{

  TAU_PROFILE_TIMER(t, "shmem_group_inquire()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_inquire( handle, sgs) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_inquire( handle, sgs) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_group_inquire_jw wrapper function 
******************************************************/
void shmem_group_inquire_jw( int handle, shmem_group_t * sgs)
{

  TAU_PROFILE_TIMER(t, "shmem_group_inquire_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_inquire_jw( handle, sgs) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_inquire_jw( handle, sgs) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_inquire_jw wrapper function 
******************************************************/
void SHMEM_GROUP_INQUIRE_JW( int handle, shmem_group_t * sgs)
{

  TAU_PROFILE_TIMER(t, "shmem_group_inquire_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_inquire_jw( handle, sgs) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_inquire_jw( handle, sgs) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_inquire_jw wrapper function 
******************************************************/
void shmem_group_inquire_jw_( int handle, shmem_group_t * sgs)
{

  TAU_PROFILE_TIMER(t, "shmem_group_inquire_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_inquire_jw( handle, sgs) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_inquire_jw( handle, sgs) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_group_inquire_jw wrapper function 
******************************************************/
void shmem_group_inquire_jw__( int handle, shmem_group_t * sgs)
{

  TAU_PROFILE_TIMER(t, "shmem_group_inquire_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_group_inquire_jw( handle, sgs) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_group_inquire_jw( handle, sgs) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmemalign wrapper function 
******************************************************/
void * shmemalign( size_t align, size_t size)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmemalign()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmemalign( align, size) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmemalign( align, size) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmemalign wrapper function 
******************************************************/
void * SHMEMALIGN( size_t align, size_t size)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmemalign()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmemalign( align, size) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmemalign( align, size) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmemalign wrapper function 
******************************************************/
void * shmemalign_( size_t align, size_t size)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmemalign()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmemalign( align, size) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmemalign( align, size) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmemalign wrapper function 
******************************************************/
void * shmemalign__( size_t align, size_t size)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmemalign()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmemalign( align, size) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmemalign( align, size) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmemalign_jw wrapper function 
******************************************************/
void * shmemalign_jw( size_t align, size_t size)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmemalign_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmemalign_jw( align, size) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmemalign_jw( align, size) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmemalign_jw wrapper function 
******************************************************/
void * SHMEMALIGN_JW( size_t align, size_t size)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmemalign_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmemalign_jw( align, size) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmemalign_jw( align, size) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmemalign_jw wrapper function 
******************************************************/
void * shmemalign_jw_( size_t align, size_t size)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmemalign_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmemalign_jw( align, size) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmemalign_jw( align, size) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmemalign_jw wrapper function 
******************************************************/
void * shmemalign_jw__( size_t align, size_t size)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmemalign_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmemalign_jw( align, size) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmemalign_jw( align, size) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


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
***      shmem_swap_jw wrapper function 
******************************************************/
long shmem_swap_jw( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap_jw wrapper function 
******************************************************/
long SHMEM_SWAP_JW( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap_jw wrapper function 
******************************************************/
long shmem_swap_jw_( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap_jw wrapper function 
******************************************************/
long shmem_swap_jw__( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap_jw( addr, value, pe) ; 
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
***      shmem_cswap_jw wrapper function 
******************************************************/
long shmem_cswap_jw( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_cswap_jw wrapper function 
******************************************************/
long SHMEM_CSWAP_JW( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_cswap_jw wrapper function 
******************************************************/
long shmem_cswap_jw_( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_cswap_jw wrapper function 
******************************************************/
long shmem_cswap_jw__( long * addr, long match, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_cswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_cswap_jw( addr, match, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_cswap_jw( addr, match, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_mswap wrapper function 
******************************************************/
long shmem_mswap( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_mswap wrapper function 
******************************************************/
long SHMEM_MSWAP( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_mswap wrapper function 
******************************************************/
long shmem_mswap_( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_mswap wrapper function 
******************************************************/
long shmem_mswap__( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_mswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_mswap( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_mswap( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_mswap_jw wrapper function 
******************************************************/
long shmem_mswap_jw( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_mswap_jw wrapper function 
******************************************************/
long SHMEM_MSWAP_JW( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_mswap_jw wrapper function 
******************************************************/
long shmem_mswap_jw_( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_mswap_jw wrapper function 
******************************************************/
long shmem_mswap_jw__( long * addr, long mask, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_mswap_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_mswap_jw( addr, mask, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_mswap_jw( addr, mask, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_inc wrapper function 
******************************************************/
void shmem_inc( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_inc wrapper function 
******************************************************/
void SHMEM_INC( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_inc wrapper function 
******************************************************/
void shmem_inc_( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_inc wrapper function 
******************************************************/
void shmem_inc__( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_inc( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_inc( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_inc_jw wrapper function 
******************************************************/
void shmem_inc_jw( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_inc_jw wrapper function 
******************************************************/
void SHMEM_INC_JW( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_inc_jw wrapper function 
******************************************************/
void shmem_inc_jw_( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_inc_jw wrapper function 
******************************************************/
void shmem_inc_jw__( long * addr, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_inc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_inc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_inc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
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
***      shmem_finc_jw wrapper function 
******************************************************/
long shmem_finc_jw( long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_finc_jw wrapper function 
******************************************************/
long SHMEM_FINC_JW( long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_finc_jw wrapper function 
******************************************************/
long shmem_finc_jw_( long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_finc_jw wrapper function 
******************************************************/
long shmem_finc_jw__( long * addr, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_finc_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_finc_jw( addr, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_finc_jw( addr, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_add wrapper function 
******************************************************/
void shmem_add( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_add wrapper function 
******************************************************/
void SHMEM_ADD( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_add wrapper function 
******************************************************/
void shmem_add_( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_add wrapper function 
******************************************************/
void shmem_add__( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_add( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_add( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_add_jw wrapper function 
******************************************************/
void shmem_add_jw( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_add_jw wrapper function 
******************************************************/
void SHMEM_ADD_JW( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_add_jw wrapper function 
******************************************************/
void shmem_add_jw_( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_add_jw wrapper function 
******************************************************/
void shmem_add_jw__( long * addr, long value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_add_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_add_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_add_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
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
***      shmem_fadd_jw wrapper function 
******************************************************/
long shmem_fadd_jw( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_fadd_jw wrapper function 
******************************************************/
long SHMEM_FADD_JW( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_fadd_jw wrapper function 
******************************************************/
long shmem_fadd_jw_( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_fadd_jw( addr, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_fadd_jw wrapper function 
******************************************************/
long shmem_fadd_jw__( long * addr, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_fadd_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_fadd_jw( addr, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_fadd_jw( addr, value, pe) ; 
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
***      shmem_broadcast_jw wrapper function 
******************************************************/
void shmem_broadcast_jw( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast_jw wrapper function 
******************************************************/
void SHMEM_BROADCAST_JW( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast_jw wrapper function 
******************************************************/
void shmem_broadcast_jw_( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast_jw wrapper function 
******************************************************/
void shmem_broadcast_jw__( void * trg, const void * src, size_t len, int peroot, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast_jw( trg, src, len, peroot, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect wrapper function 
******************************************************/
void shmem_collect( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect wrapper function 
******************************************************/
void SHMEM_COLLECT( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect wrapper function 
******************************************************/
void shmem_collect_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect wrapper function 
******************************************************/
void shmem_collect__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect_jw wrapper function 
******************************************************/
void shmem_collect_jw( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect_jw wrapper function 
******************************************************/
void SHMEM_COLLECT_JW( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect_jw wrapper function 
******************************************************/
void shmem_collect_jw_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect_jw wrapper function 
******************************************************/
void shmem_collect_jw__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect wrapper function 
******************************************************/
void shmem_fcollect( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect wrapper function 
******************************************************/
void SHMEM_FCOLLECT( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect wrapper function 
******************************************************/
void shmem_fcollect_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect wrapper function 
******************************************************/
void shmem_fcollect__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect_jw wrapper function 
******************************************************/
void shmem_fcollect_jw( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect_jw wrapper function 
******************************************************/
void SHMEM_FCOLLECT_JW( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect_jw wrapper function 
******************************************************/
void shmem_fcollect_jw_( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect_jw wrapper function 
******************************************************/
void shmem_fcollect_jw__( void * trg, const void * src, size_t len, int pestart, int log_pestride, int pesize, long * psync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect_jw( trg, src, len, pestart, log_pestride, pesize, psync) ; 
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
***      shmem_get_jw wrapper function 
******************************************************/
void shmem_get_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get_jw wrapper function 
******************************************************/
void SHMEM_GET_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get_jw wrapper function 
******************************************************/
void shmem_get_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get_jw wrapper function 
******************************************************/
void shmem_get_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get_jw( trg, src, len, pe) ; 
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
***      shmem_put_jw wrapper function 
******************************************************/
void shmem_put_jw( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put_jw wrapper function 
******************************************************/
void SHMEM_PUT_JW( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put_jw wrapper function 
******************************************************/
void shmem_put_jw_( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put_jw( trg, src, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put_jw wrapper function 
******************************************************/
void shmem_put_jw__( void * trg, const void * src, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put_jw( trg, src, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put_jw( trg, src, len, pe) ; 
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
***      shmem_iget_jw wrapper function 
******************************************************/
void shmem_iget_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget_jw wrapper function 
******************************************************/
void SHMEM_IGET_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget_jw wrapper function 
******************************************************/
void shmem_iget_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget_jw wrapper function 
******************************************************/
void shmem_iget_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget_jw( trg, src, tst, sst, len, pe) ; 
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
***      shmem_iput_jw wrapper function 
******************************************************/
void shmem_iput_jw( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput_jw wrapper function 
******************************************************/
void SHMEM_IPUT_JW( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput_jw wrapper function 
******************************************************/
void shmem_iput_jw_( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput_jw wrapper function 
******************************************************/
void shmem_iput_jw__( void * trg, const void * src, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput_jw( trg, src, tst, sst, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput_jw( trg, src, tst, sst, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget wrapper function 
******************************************************/
void shmem_ixget( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget wrapper function 
******************************************************/
void SHMEM_IXGET( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget wrapper function 
******************************************************/
void shmem_ixget_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget wrapper function 
******************************************************/
void shmem_ixget__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget_jw wrapper function 
******************************************************/
void shmem_ixget_jw( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget_jw wrapper function 
******************************************************/
void SHMEM_IXGET_JW( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget_jw wrapper function 
******************************************************/
void shmem_ixget_jw_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget_jw wrapper function 
******************************************************/
void shmem_ixget_jw__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput wrapper function 
******************************************************/
void shmem_ixput( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput wrapper function 
******************************************************/
void SHMEM_IXPUT( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput wrapper function 
******************************************************/
void shmem_ixput_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput wrapper function 
******************************************************/
void shmem_ixput__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput_jw wrapper function 
******************************************************/
void shmem_ixput_jw( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput_jw wrapper function 
******************************************************/
void SHMEM_IXPUT_JW( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput_jw wrapper function 
******************************************************/
void shmem_ixput_jw_( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput_jw( trg, src, idx, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput_jw wrapper function 
******************************************************/
void shmem_ixput_jw__( void * trg, const void * src, long * idx, size_t len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput_jw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput_jw( trg, src, idx, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput_jw( trg, src, idx, len, pe) ; 
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

