#include <TAU.h>
#include <shmem.h>
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
***      shmem_version wrapper function 
******************************************************/
void shmem_version( )
{

  TAU_PROFILE_TIMER(t, "shmem_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_version( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_version( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_version wrapper function 
******************************************************/
void SHMEM_VERSION( )
{

  TAU_PROFILE_TIMER(t, "shmem_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_version( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_version( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_version wrapper function 
******************************************************/
void shmem_version_( )
{

  TAU_PROFILE_TIMER(t, "shmem_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_version( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_version( ) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_version wrapper function 
******************************************************/
void shmem_version__( )
{

  TAU_PROFILE_TIMER(t, "shmem_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_version( ) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_version( ) ; 
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
***      shmem_set_cache_line_inv wrapper function 
******************************************************/
void shmem_set_cache_line_inv( long * target)
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_line_inv( target) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_line_inv( target) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_line_inv wrapper function 
******************************************************/
void SHMEM_SET_CACHE_LINE_INV( long * target)
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_line_inv( target) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_line_inv( target) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_line_inv wrapper function 
******************************************************/
void shmem_set_cache_line_inv_( long * target)
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_line_inv( target) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_line_inv( target) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_set_cache_line_inv wrapper function 
******************************************************/
void shmem_set_cache_line_inv__( long * target)
{

  TAU_PROFILE_TIMER(t, "shmem_set_cache_line_inv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_set_cache_line_inv( target) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_set_cache_line_inv( target) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
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
void shmem_udcflush_line( long * target)
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush_line()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush_line( target) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush_line( target) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush_line wrapper function 
******************************************************/
void SHMEM_UDCFLUSH_LINE( long * target)
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush_line()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush_line( target) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush_line( target) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush_line wrapper function 
******************************************************/
void shmem_udcflush_line_( long * target)
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush_line()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush_line( target) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush_line( target) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_udcflush_line wrapper function 
******************************************************/
void shmem_udcflush_line__( long * target)
{

  TAU_PROFILE_TIMER(t, "shmem_udcflush_line()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_udcflush_line( target) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_udcflush_line( target) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_stack wrapper function 
******************************************************/
void shmem_stack( void * stack_var)
{

  TAU_PROFILE_TIMER(t, "shmem_stack()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_stack( stack_var) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_stack( stack_var) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_stack wrapper function 
******************************************************/
void SHMEM_STACK( void * stack_var)
{

  TAU_PROFILE_TIMER(t, "shmem_stack()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_stack( stack_var) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_stack( stack_var) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_stack wrapper function 
******************************************************/
void shmem_stack_( void * stack_var)
{

  TAU_PROFILE_TIMER(t, "shmem_stack()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_stack( stack_var) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_stack( stack_var) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_stack wrapper function 
******************************************************/
void shmem_stack__( void * stack_var)
{

  TAU_PROFILE_TIMER(t, "shmem_stack()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_stack( stack_var) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_stack( stack_var) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ptr wrapper function 
******************************************************/
void * shmem_ptr( void * target, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr wrapper function 
******************************************************/
void * SHMEM_PTR( void * target, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr wrapper function 
******************************************************/
void * shmem_ptr_( void * target, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_ptr wrapper function 
******************************************************/
void * shmem_ptr__( void * target, int pe)
{
  void * retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_ptr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_ptr( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_ptr( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_start_pes wrapper function 
******************************************************/
void shmem_start_pes( int npes)
{

  TAU_PROFILE_TIMER(t, "shmem_start_pes()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_start_pes( npes) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_start_pes( npes) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_start_pes wrapper function 
******************************************************/
void SHMEM_START_PES( int npes)
{

  TAU_PROFILE_TIMER(t, "shmem_start_pes()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_start_pes( npes) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_start_pes( npes) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_start_pes wrapper function 
******************************************************/
void shmem_start_pes_( int npes)
{

  TAU_PROFILE_TIMER(t, "shmem_start_pes()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_start_pes( npes) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_start_pes( npes) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_start_pes wrapper function 
******************************************************/
void shmem_start_pes__( int npes)
{

  TAU_PROFILE_TIMER(t, "shmem_start_pes()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_start_pes( npes) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_start_pes( npes) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
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
***      shmem_barrier wrapper function 
******************************************************/
void shmem_barrier( int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier( PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier( PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier wrapper function 
******************************************************/
void SHMEM_BARRIER( int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier( PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier( PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier wrapper function 
******************************************************/
void shmem_barrier_( int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier( PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier( PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_barrier wrapper function 
******************************************************/
void shmem_barrier__( int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_barrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_barrier( PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_barrier( PE_start, logPE_stride, PE_size, pSync) ; 
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
***      shmem_broadcast wrapper function 
******************************************************/
void shmem_broadcast( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast wrapper function 
******************************************************/
void SHMEM_BROADCAST( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast wrapper function 
******************************************************/
void shmem_broadcast_( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast wrapper function 
******************************************************/
void shmem_broadcast__( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_broadcast32 wrapper function 
******************************************************/
void shmem_broadcast32( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32 wrapper function 
******************************************************/
void SHMEM_BROADCAST32( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32 wrapper function 
******************************************************/
void shmem_broadcast32_( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast32 wrapper function 
******************************************************/
void shmem_broadcast32__( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast32( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast32( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_broadcast64 wrapper function 
******************************************************/
void shmem_broadcast64( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64 wrapper function 
******************************************************/
void SHMEM_BROADCAST64( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64 wrapper function 
******************************************************/
void shmem_broadcast64_( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast64 wrapper function 
******************************************************/
void shmem_broadcast64__( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast64( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast64( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_broadcast128 wrapper function 
******************************************************/
void shmem_broadcast128( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast128( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast128( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast128 wrapper function 
******************************************************/
void SHMEM_BROADCAST128( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast128( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast128( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast128 wrapper function 
******************************************************/
void shmem_broadcast128_( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast128( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast128( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_broadcast128 wrapper function 
******************************************************/
void shmem_broadcast128__( void * target, void * source, int nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_broadcast128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_broadcast128( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_broadcast128( target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect wrapper function 
******************************************************/
void shmem_fcollect( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect wrapper function 
******************************************************/
void SHMEM_FCOLLECT( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect wrapper function 
******************************************************/
void shmem_fcollect_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect wrapper function 
******************************************************/
void shmem_fcollect__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect32 wrapper function 
******************************************************/
void shmem_fcollect32( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32 wrapper function 
******************************************************/
void SHMEM_FCOLLECT32( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32 wrapper function 
******************************************************/
void shmem_fcollect32_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect32 wrapper function 
******************************************************/
void shmem_fcollect32__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect64 wrapper function 
******************************************************/
void shmem_fcollect64( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64 wrapper function 
******************************************************/
void SHMEM_FCOLLECT64( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64 wrapper function 
******************************************************/
void shmem_fcollect64_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect64 wrapper function 
******************************************************/
void shmem_fcollect64__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_fcollect128 wrapper function 
******************************************************/
void shmem_fcollect128( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect128 wrapper function 
******************************************************/
void SHMEM_FCOLLECT128( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect128 wrapper function 
******************************************************/
void shmem_fcollect128_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_fcollect128 wrapper function 
******************************************************/
void shmem_fcollect128__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_fcollect128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_fcollect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_fcollect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect wrapper function 
******************************************************/
void shmem_collect( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect wrapper function 
******************************************************/
void SHMEM_COLLECT( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect wrapper function 
******************************************************/
void shmem_collect_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect wrapper function 
******************************************************/
void shmem_collect__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect32 wrapper function 
******************************************************/
void shmem_collect32( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32 wrapper function 
******************************************************/
void SHMEM_COLLECT32( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32 wrapper function 
******************************************************/
void shmem_collect32_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect32 wrapper function 
******************************************************/
void shmem_collect32__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect64 wrapper function 
******************************************************/
void shmem_collect64( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64 wrapper function 
******************************************************/
void SHMEM_COLLECT64( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64 wrapper function 
******************************************************/
void shmem_collect64_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect64 wrapper function 
******************************************************/
void shmem_collect64__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_collect128 wrapper function 
******************************************************/
void shmem_collect128( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect128 wrapper function 
******************************************************/
void SHMEM_COLLECT128( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect128 wrapper function 
******************************************************/
void shmem_collect128_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_collect128 wrapper function 
******************************************************/
void shmem_collect128__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_collect128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_collect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_collect128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_getmem wrapper function 
******************************************************/
void shmem_getmem( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem wrapper function 
******************************************************/
void SHMEM_GETMEM( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem wrapper function 
******************************************************/
void shmem_getmem_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_getmem wrapper function 
******************************************************/
void shmem_getmem__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_getmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_getmem( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_getmem( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_putmem wrapper function 
******************************************************/
void shmem_putmem( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem wrapper function 
******************************************************/
void SHMEM_PUTMEM( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem wrapper function 
******************************************************/
void shmem_putmem_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_putmem wrapper function 
******************************************************/
void shmem_putmem__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_putmem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_putmem( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_putmem( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_get wrapper function 
******************************************************/
void shmem_short_get( short * target, short * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get wrapper function 
******************************************************/
void SHMEM_SHORT_GET( short * target, short * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get wrapper function 
******************************************************/
void shmem_short_get_( short * target, short * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_get wrapper function 
******************************************************/
void shmem_short_get__( short * target, short * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_put wrapper function 
******************************************************/
void shmem_short_put( short * target, short * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put wrapper function 
******************************************************/
void SHMEM_SHORT_PUT( short * target, short * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put wrapper function 
******************************************************/
void shmem_short_put_( short * target, short * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_put wrapper function 
******************************************************/
void shmem_short_put__( short * target, short * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_get wrapper function 
******************************************************/
void shmem_int_get( int * target, int * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get wrapper function 
******************************************************/
void SHMEM_INT_GET( int * target, int * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get wrapper function 
******************************************************/
void shmem_int_get_( int * target, int * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_get wrapper function 
******************************************************/
void shmem_int_get__( int * target, int * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_put wrapper function 
******************************************************/
void shmem_int_put( int * target, int * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put wrapper function 
******************************************************/
void SHMEM_INT_PUT( int * target, int * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put wrapper function 
******************************************************/
void shmem_int_put_( int * target, int * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_put wrapper function 
******************************************************/
void shmem_int_put__( int * target, int * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_get wrapper function 
******************************************************/
void shmem_long_get( long * target, long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get wrapper function 
******************************************************/
void SHMEM_LONG_GET( long * target, long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get wrapper function 
******************************************************/
void shmem_long_get_( long * target, long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_get wrapper function 
******************************************************/
void shmem_long_get__( long * target, long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_put wrapper function 
******************************************************/
void shmem_long_put( long * target, long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put wrapper function 
******************************************************/
void SHMEM_LONG_PUT( long * target, long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put wrapper function 
******************************************************/
void shmem_long_put_( long * target, long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_put wrapper function 
******************************************************/
void shmem_long_put__( long * target, long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_get wrapper function 
******************************************************/
void shmem_longlong_get( long long * target, long long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get wrapper function 
******************************************************/
void SHMEM_LONGLONG_GET( long long * target, long long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get wrapper function 
******************************************************/
void shmem_longlong_get_( long long * target, long long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_get wrapper function 
******************************************************/
void shmem_longlong_get__( long long * target, long long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_put wrapper function 
******************************************************/
void shmem_longlong_put( long long * target, long long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put wrapper function 
******************************************************/
void SHMEM_LONGLONG_PUT( long long * target, long long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put wrapper function 
******************************************************/
void shmem_longlong_put_( long long * target, long long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_put wrapper function 
******************************************************/
void shmem_longlong_put__( long long * target, long long * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_get wrapper function 
******************************************************/
void shmem_float_get( float * target, float * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get wrapper function 
******************************************************/
void SHMEM_FLOAT_GET( float * target, float * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get wrapper function 
******************************************************/
void shmem_float_get_( float * target, float * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_get wrapper function 
******************************************************/
void shmem_float_get__( float * target, float * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_put wrapper function 
******************************************************/
void shmem_float_put( float * target, float * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put wrapper function 
******************************************************/
void SHMEM_FLOAT_PUT( float * target, float * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put wrapper function 
******************************************************/
void shmem_float_put_( float * target, float * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_put wrapper function 
******************************************************/
void shmem_float_put__( float * target, float * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_get wrapper function 
******************************************************/
void shmem_double_get( double * target, double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get wrapper function 
******************************************************/
void SHMEM_DOUBLE_GET( double * target, double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get wrapper function 
******************************************************/
void shmem_double_get_( double * target, double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_get wrapper function 
******************************************************/
void shmem_double_get__( double * target, double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_put wrapper function 
******************************************************/
void shmem_double_put( double * target, double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put wrapper function 
******************************************************/
void SHMEM_DOUBLE_PUT( double * target, double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put wrapper function 
******************************************************/
void shmem_double_put_( double * target, double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_put wrapper function 
******************************************************/
void shmem_double_put__( double * target, double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_get wrapper function 
******************************************************/
void shmem_longdouble_get( long double * target, long double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_GET( long double * target, long double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get wrapper function 
******************************************************/
void shmem_longdouble_get_( long double * target, long double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_get wrapper function 
******************************************************/
void shmem_longdouble_get__( long double * target, long double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_put wrapper function 
******************************************************/
void shmem_longdouble_put( long double * target, long double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_PUT( long double * target, long double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put wrapper function 
******************************************************/
void shmem_longdouble_put_( long double * target, long double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_put wrapper function 
******************************************************/
void shmem_longdouble_put__( long double * target, long double * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_get wrapper function 
******************************************************/
void shmem_complexf_get( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_get wrapper function 
******************************************************/
void SHMEM_COMPLEXF_GET( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_get wrapper function 
******************************************************/
void shmem_complexf_get_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_get wrapper function 
******************************************************/
void shmem_complexf_get__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_put wrapper function 
******************************************************/
void shmem_complexf_put( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_put wrapper function 
******************************************************/
void SHMEM_COMPLEXF_PUT( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_put wrapper function 
******************************************************/
void shmem_complexf_put_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_put wrapper function 
******************************************************/
void shmem_complexf_put__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_get wrapper function 
******************************************************/
void shmem_complexd_get( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_get wrapper function 
******************************************************/
void SHMEM_COMPLEXD_GET( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_get wrapper function 
******************************************************/
void shmem_complexd_get_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_get wrapper function 
******************************************************/
void shmem_complexd_get__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_put wrapper function 
******************************************************/
void shmem_complexd_put( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_put wrapper function 
******************************************************/
void SHMEM_COMPLEXD_PUT( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_put wrapper function 
******************************************************/
void shmem_complexd_put_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_put wrapper function 
******************************************************/
void shmem_complexd_put__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get wrapper function 
******************************************************/
void shmem_get( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get wrapper function 
******************************************************/
void SHMEM_GET( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get wrapper function 
******************************************************/
void shmem_get_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get wrapper function 
******************************************************/
void shmem_get__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put wrapper function 
******************************************************/
void shmem_put( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put wrapper function 
******************************************************/
void SHMEM_PUT( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put wrapper function 
******************************************************/
void shmem_put_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put wrapper function 
******************************************************/
void shmem_put__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get32 wrapper function 
******************************************************/
void shmem_get32( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32 wrapper function 
******************************************************/
void SHMEM_GET32( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32 wrapper function 
******************************************************/
void shmem_get32_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get32 wrapper function 
******************************************************/
void shmem_get32__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get32( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get32( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put32 wrapper function 
******************************************************/
void shmem_put32( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32 wrapper function 
******************************************************/
void SHMEM_PUT32( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32 wrapper function 
******************************************************/
void shmem_put32_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put32 wrapper function 
******************************************************/
void shmem_put32__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put32( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put32( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get64 wrapper function 
******************************************************/
void shmem_get64( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64 wrapper function 
******************************************************/
void SHMEM_GET64( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64 wrapper function 
******************************************************/
void shmem_get64_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get64 wrapper function 
******************************************************/
void shmem_get64__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get64( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get64( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put64 wrapper function 
******************************************************/
void shmem_put64( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64 wrapper function 
******************************************************/
void SHMEM_PUT64( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64 wrapper function 
******************************************************/
void shmem_put64_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put64 wrapper function 
******************************************************/
void shmem_put64__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put64( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put64( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_get128 wrapper function 
******************************************************/
void shmem_get128( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128 wrapper function 
******************************************************/
void SHMEM_GET128( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128 wrapper function 
******************************************************/
void shmem_get128_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_get128 wrapper function 
******************************************************/
void shmem_get128__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_get128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_get128( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_get128( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_put128 wrapper function 
******************************************************/
void shmem_put128( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128 wrapper function 
******************************************************/
void SHMEM_PUT128( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128 wrapper function 
******************************************************/
void shmem_put128_( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128( target, source, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_put128 wrapper function 
******************************************************/
void shmem_put128__( void * target, void * source, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_put128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_put128( target, source, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_put128( target, source, len, pe) ; 
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
***      shmem_short_g wrapper function 
******************************************************/
short shmem_short_g( short * addr, int pe)
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
short SHMEM_SHORT_G( short * addr, int pe)
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
short shmem_short_g_( short * addr, int pe)
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
short shmem_short_g__( short * addr, int pe)
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
int shmem_int_g( int * addr, int pe)
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
int SHMEM_INT_G( int * addr, int pe)
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
int shmem_int_g_( int * addr, int pe)
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
int shmem_int_g__( int * addr, int pe)
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
long shmem_long_g( long * addr, int pe)
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
long SHMEM_LONG_G( long * addr, int pe)
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
long shmem_long_g_( long * addr, int pe)
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
long shmem_long_g__( long * addr, int pe)
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
long long shmem_longlong_g( long long * addr, int pe)
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
long long SHMEM_LONGLONG_G( long long * addr, int pe)
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
long long shmem_longlong_g_( long long * addr, int pe)
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
long long shmem_longlong_g__( long long * addr, int pe)
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
float shmem_float_g( float * addr, int pe)
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
float SHMEM_FLOAT_G( float * addr, int pe)
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
float shmem_float_g_( float * addr, int pe)
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
float shmem_float_g__( float * addr, int pe)
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
double shmem_double_g( double * addr, int pe)
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
double SHMEM_DOUBLE_G( double * addr, int pe)
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
double shmem_double_g_( double * addr, int pe)
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
double shmem_double_g__( double * addr, int pe)
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
***      shmem_longdouble_g wrapper function 
******************************************************/
long double shmem_longdouble_g( long double * addr, int pe)
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
long double SHMEM_LONGDOUBLE_G( long double * addr, int pe)
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
long double shmem_longdouble_g_( long double * addr, int pe)
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
long double shmem_longdouble_g__( long double * addr, int pe)
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
***      shmem_iget wrapper function 
******************************************************/
void shmem_iget( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget wrapper function 
******************************************************/
void SHMEM_IGET( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget wrapper function 
******************************************************/
void shmem_iget_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget wrapper function 
******************************************************/
void shmem_iget__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iget32 wrapper function 
******************************************************/
void shmem_iget32( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32 wrapper function 
******************************************************/
void SHMEM_IGET32( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32 wrapper function 
******************************************************/
void shmem_iget32_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget32 wrapper function 
******************************************************/
void shmem_iget32__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget32( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget32( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iget64 wrapper function 
******************************************************/
void shmem_iget64( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64 wrapper function 
******************************************************/
void SHMEM_IGET64( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64 wrapper function 
******************************************************/
void shmem_iget64_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget64 wrapper function 
******************************************************/
void shmem_iget64__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget64( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget64( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iget128 wrapper function 
******************************************************/
void shmem_iget128( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128 wrapper function 
******************************************************/
void SHMEM_IGET128( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128 wrapper function 
******************************************************/
void shmem_iget128_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iget128 wrapper function 
******************************************************/
void shmem_iget128__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iget128( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iget128( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_iget wrapper function 
******************************************************/
void shmem_short_iget( short * target, short * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget wrapper function 
******************************************************/
void SHMEM_SHORT_IGET( short * target, short * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget wrapper function 
******************************************************/
void shmem_short_iget_( short * target, short * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iget wrapper function 
******************************************************/
void shmem_short_iget__( short * target, short * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_iget wrapper function 
******************************************************/
void shmem_int_iget( int * target, int * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget wrapper function 
******************************************************/
void SHMEM_INT_IGET( int * target, int * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget wrapper function 
******************************************************/
void shmem_int_iget_( int * target, int * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iget wrapper function 
******************************************************/
void shmem_int_iget__( int * target, int * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_iget wrapper function 
******************************************************/
void shmem_long_iget( long * target, long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget wrapper function 
******************************************************/
void SHMEM_LONG_IGET( long * target, long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget wrapper function 
******************************************************/
void shmem_long_iget_( long * target, long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iget wrapper function 
******************************************************/
void shmem_long_iget__( long * target, long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_iget wrapper function 
******************************************************/
void shmem_longlong_iget( long long * target, long long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget wrapper function 
******************************************************/
void SHMEM_LONGLONG_IGET( long long * target, long long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget wrapper function 
******************************************************/
void shmem_longlong_iget_( long long * target, long long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iget wrapper function 
******************************************************/
void shmem_longlong_iget__( long long * target, long long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_iget wrapper function 
******************************************************/
void shmem_float_iget( float * target, float * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget wrapper function 
******************************************************/
void SHMEM_FLOAT_IGET( float * target, float * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget wrapper function 
******************************************************/
void shmem_float_iget_( float * target, float * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iget wrapper function 
******************************************************/
void shmem_float_iget__( float * target, float * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_iget wrapper function 
******************************************************/
void shmem_double_iget( double * target, double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget wrapper function 
******************************************************/
void SHMEM_DOUBLE_IGET( double * target, double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget wrapper function 
******************************************************/
void shmem_double_iget_( double * target, double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iget wrapper function 
******************************************************/
void shmem_double_iget__( double * target, double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_iget wrapper function 
******************************************************/
void shmem_longdouble_iget( long double * target, long double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_IGET( long double * target, long double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget wrapper function 
******************************************************/
void shmem_longdouble_iget_( long double * target, long double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iget wrapper function 
******************************************************/
void shmem_longdouble_iget__( long double * target, long double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_iget wrapper function 
******************************************************/
void shmem_complexf_iget( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_iget wrapper function 
******************************************************/
void SHMEM_COMPLEXF_IGET( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_iget wrapper function 
******************************************************/
void shmem_complexf_iget_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_iget wrapper function 
******************************************************/
void shmem_complexf_iget__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_iget wrapper function 
******************************************************/
void shmem_complexd_iget( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_iget wrapper function 
******************************************************/
void SHMEM_COMPLEXD_IGET( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_iget wrapper function 
******************************************************/
void shmem_complexd_iget_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_iget wrapper function 
******************************************************/
void shmem_complexd_iget__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_iget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_iget( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_iget( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput wrapper function 
******************************************************/
void shmem_iput( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput wrapper function 
******************************************************/
void SHMEM_IPUT( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput wrapper function 
******************************************************/
void shmem_iput_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput wrapper function 
******************************************************/
void shmem_iput__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput32 wrapper function 
******************************************************/
void shmem_iput32( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32 wrapper function 
******************************************************/
void SHMEM_IPUT32( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32 wrapper function 
******************************************************/
void shmem_iput32_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput32 wrapper function 
******************************************************/
void shmem_iput32__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput32( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput32( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput64 wrapper function 
******************************************************/
void shmem_iput64( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64 wrapper function 
******************************************************/
void SHMEM_IPUT64( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64 wrapper function 
******************************************************/
void shmem_iput64_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput64 wrapper function 
******************************************************/
void shmem_iput64__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput64( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput64( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_iput128 wrapper function 
******************************************************/
void shmem_iput128( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128 wrapper function 
******************************************************/
void SHMEM_IPUT128( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128 wrapper function 
******************************************************/
void shmem_iput128_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_iput128 wrapper function 
******************************************************/
void shmem_iput128__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_iput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_iput128( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_iput128( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_iput wrapper function 
******************************************************/
void shmem_short_iput( short * target, short * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput wrapper function 
******************************************************/
void SHMEM_SHORT_IPUT( short * target, short * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput wrapper function 
******************************************************/
void shmem_short_iput_( short * target, short * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_iput wrapper function 
******************************************************/
void shmem_short_iput__( short * target, short * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_iput wrapper function 
******************************************************/
void shmem_int_iput( int * target, int * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput wrapper function 
******************************************************/
void SHMEM_INT_IPUT( int * target, int * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput wrapper function 
******************************************************/
void shmem_int_iput_( int * target, int * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_iput wrapper function 
******************************************************/
void shmem_int_iput__( int * target, int * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_iput wrapper function 
******************************************************/
void shmem_long_iput( long * target, long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput wrapper function 
******************************************************/
void SHMEM_LONG_IPUT( long * target, long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput wrapper function 
******************************************************/
void shmem_long_iput_( long * target, long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_iput wrapper function 
******************************************************/
void shmem_long_iput__( long * target, long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_iput wrapper function 
******************************************************/
void shmem_longlong_iput( long long * target, long long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput wrapper function 
******************************************************/
void SHMEM_LONGLONG_IPUT( long long * target, long long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput wrapper function 
******************************************************/
void shmem_longlong_iput_( long long * target, long long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_iput wrapper function 
******************************************************/
void shmem_longlong_iput__( long long * target, long long * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_iput wrapper function 
******************************************************/
void shmem_float_iput( float * target, float * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput wrapper function 
******************************************************/
void SHMEM_FLOAT_IPUT( float * target, float * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput wrapper function 
******************************************************/
void shmem_float_iput_( float * target, float * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_iput wrapper function 
******************************************************/
void shmem_float_iput__( float * target, float * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_iput wrapper function 
******************************************************/
void shmem_double_iput( double * target, double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput wrapper function 
******************************************************/
void SHMEM_DOUBLE_IPUT( double * target, double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput wrapper function 
******************************************************/
void shmem_double_iput_( double * target, double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_iput wrapper function 
******************************************************/
void shmem_double_iput__( double * target, double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_iput wrapper function 
******************************************************/
void shmem_longdouble_iput( long double * target, long double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_IPUT( long double * target, long double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput wrapper function 
******************************************************/
void shmem_longdouble_iput_( long double * target, long double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_iput wrapper function 
******************************************************/
void shmem_longdouble_iput__( long double * target, long double * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_iput wrapper function 
******************************************************/
void shmem_complexf_iput( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_iput wrapper function 
******************************************************/
void SHMEM_COMPLEXF_IPUT( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_iput wrapper function 
******************************************************/
void shmem_complexf_iput_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_iput wrapper function 
******************************************************/
void shmem_complexf_iput__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_iput wrapper function 
******************************************************/
void shmem_complexd_iput( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_iput wrapper function 
******************************************************/
void SHMEM_COMPLEXD_IPUT( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_iput wrapper function 
******************************************************/
void shmem_complexd_iput_( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_iput wrapper function 
******************************************************/
void shmem_complexd_iput__( void * target, void * source, int target_inc, int source_inc, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_iput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_iput( target, source, target_inc, source_inc, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_iput( target, source, target_inc, source_inc, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget wrapper function 
******************************************************/
void shmem_ixget( void * target, void * source, long long * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget wrapper function 
******************************************************/
void SHMEM_IXGET( void * target, void * source, long long * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget wrapper function 
******************************************************/
void shmem_ixget_( void * target, void * source, long long * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget wrapper function 
******************************************************/
void shmem_ixget__( void * target, void * source, long long * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget32 wrapper function 
******************************************************/
void shmem_ixget32( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32 wrapper function 
******************************************************/
void SHMEM_IXGET32( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32 wrapper function 
******************************************************/
void shmem_ixget32_( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget32 wrapper function 
******************************************************/
void shmem_ixget32__( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget32( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget32( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget64 wrapper function 
******************************************************/
void shmem_ixget64( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64 wrapper function 
******************************************************/
void SHMEM_IXGET64( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64 wrapper function 
******************************************************/
void shmem_ixget64_( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget64 wrapper function 
******************************************************/
void shmem_ixget64__( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget64( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget64( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixget128 wrapper function 
******************************************************/
void shmem_ixget128( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget128( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget128( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget128 wrapper function 
******************************************************/
void SHMEM_IXGET128( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget128( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget128( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget128 wrapper function 
******************************************************/
void shmem_ixget128_( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget128( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget128( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixget128 wrapper function 
******************************************************/
void shmem_ixget128__( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixget128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixget128( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixget128( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_ixget wrapper function 
******************************************************/
void shmem_short_ixget( short * target, short * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_ixget wrapper function 
******************************************************/
void SHMEM_SHORT_IXGET( short * target, short * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_ixget wrapper function 
******************************************************/
void shmem_short_ixget_( short * target, short * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_ixget wrapper function 
******************************************************/
void shmem_short_ixget__( short * target, short * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_ixget wrapper function 
******************************************************/
void shmem_int_ixget( int * target, int * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_ixget wrapper function 
******************************************************/
void SHMEM_INT_IXGET( int * target, int * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_ixget wrapper function 
******************************************************/
void shmem_int_ixget_( int * target, int * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_ixget wrapper function 
******************************************************/
void shmem_int_ixget__( int * target, int * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_ixget wrapper function 
******************************************************/
void shmem_long_ixget( long * target, long * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_ixget wrapper function 
******************************************************/
void SHMEM_LONG_IXGET( long * target, long * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_ixget wrapper function 
******************************************************/
void shmem_long_ixget_( long * target, long * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_ixget wrapper function 
******************************************************/
void shmem_long_ixget__( long * target, long * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_ixget wrapper function 
******************************************************/
void shmem_longlong_ixget( long long * target, long long * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_ixget wrapper function 
******************************************************/
void SHMEM_LONGLONG_IXGET( long long * target, long long * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_ixget wrapper function 
******************************************************/
void shmem_longlong_ixget_( long long * target, long long * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_ixget wrapper function 
******************************************************/
void shmem_longlong_ixget__( long long * target, long long * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_ixget wrapper function 
******************************************************/
void shmem_float_ixget( float * target, float * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_ixget wrapper function 
******************************************************/
void SHMEM_FLOAT_IXGET( float * target, float * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_ixget wrapper function 
******************************************************/
void shmem_float_ixget_( float * target, float * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_ixget wrapper function 
******************************************************/
void shmem_float_ixget__( float * target, float * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_ixget wrapper function 
******************************************************/
void shmem_double_ixget( double * target, double * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_ixget wrapper function 
******************************************************/
void SHMEM_DOUBLE_IXGET( double * target, double * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_ixget wrapper function 
******************************************************/
void shmem_double_ixget_( double * target, double * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_ixget wrapper function 
******************************************************/
void shmem_double_ixget__( double * target, double * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_ixget wrapper function 
******************************************************/
void shmem_longdouble_ixget( long double * target, long double * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_ixget wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_IXGET( long double * target, long double * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_ixget wrapper function 
******************************************************/
void shmem_longdouble_ixget_( long double * target, long double * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_ixget wrapper function 
******************************************************/
void shmem_longdouble_ixget__( long double * target, long double * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_ixget wrapper function 
******************************************************/
void shmem_complexf_ixget( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_ixget wrapper function 
******************************************************/
void SHMEM_COMPLEXF_IXGET( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_ixget wrapper function 
******************************************************/
void shmem_complexf_ixget_( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_ixget wrapper function 
******************************************************/
void shmem_complexf_ixget__( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_ixget wrapper function 
******************************************************/
void shmem_complexd_ixget( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_ixget wrapper function 
******************************************************/
void SHMEM_COMPLEXD_IXGET( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_ixget wrapper function 
******************************************************/
void shmem_complexd_ixget_( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_ixget wrapper function 
******************************************************/
void shmem_complexd_ixget__( void * target, void * source, int * source_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_ixget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_ixget( target, source, source_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_ixget( target, source, source_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput wrapper function 
******************************************************/
void shmem_ixput( void * target, void * source, long long * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput wrapper function 
******************************************************/
void SHMEM_IXPUT( void * target, void * source, long long * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput wrapper function 
******************************************************/
void shmem_ixput_( void * target, void * source, long long * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput wrapper function 
******************************************************/
void shmem_ixput__( void * target, void * source, long long * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput32 wrapper function 
******************************************************/
void shmem_ixput32( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32 wrapper function 
******************************************************/
void SHMEM_IXPUT32( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32 wrapper function 
******************************************************/
void shmem_ixput32_( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput32 wrapper function 
******************************************************/
void shmem_ixput32__( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput32( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput32( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput64 wrapper function 
******************************************************/
void shmem_ixput64( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64 wrapper function 
******************************************************/
void SHMEM_IXPUT64( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64 wrapper function 
******************************************************/
void shmem_ixput64_( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput64 wrapper function 
******************************************************/
void shmem_ixput64__( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput64( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput64( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_ixput128 wrapper function 
******************************************************/
void shmem_ixput128( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput128( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput128( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput128 wrapper function 
******************************************************/
void SHMEM_IXPUT128( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput128( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput128( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput128 wrapper function 
******************************************************/
void shmem_ixput128_( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput128( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput128( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_ixput128 wrapper function 
******************************************************/
void shmem_ixput128__( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_ixput128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_ixput128( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_ixput128( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_ixput wrapper function 
******************************************************/
void shmem_short_ixput( short * target, short * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_ixput wrapper function 
******************************************************/
void SHMEM_SHORT_IXPUT( short * target, short * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_ixput wrapper function 
******************************************************/
void shmem_short_ixput_( short * target, short * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_ixput wrapper function 
******************************************************/
void shmem_short_ixput__( short * target, short * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_ixput wrapper function 
******************************************************/
void shmem_int_ixput( int * target, int * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_ixput wrapper function 
******************************************************/
void SHMEM_INT_IXPUT( int * target, int * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_ixput wrapper function 
******************************************************/
void shmem_int_ixput_( int * target, int * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_ixput wrapper function 
******************************************************/
void shmem_int_ixput__( int * target, int * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_ixput wrapper function 
******************************************************/
void shmem_long_ixput( long * target, long * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_ixput wrapper function 
******************************************************/
void SHMEM_LONG_IXPUT( long * target, long * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_ixput wrapper function 
******************************************************/
void shmem_long_ixput_( long * target, long * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_ixput wrapper function 
******************************************************/
void shmem_long_ixput__( long * target, long * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_long_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_ixput wrapper function 
******************************************************/
void shmem_longlong_ixput( long long * target, long long * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_ixput wrapper function 
******************************************************/
void SHMEM_LONGLONG_IXPUT( long long * target, long long * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_ixput wrapper function 
******************************************************/
void shmem_longlong_ixput_( long long * target, long long * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_ixput wrapper function 
******************************************************/
void shmem_longlong_ixput__( long long * target, long long * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_ixput wrapper function 
******************************************************/
void shmem_float_ixput( float * target, float * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_ixput wrapper function 
******************************************************/
void SHMEM_FLOAT_IXPUT( float * target, float * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_ixput wrapper function 
******************************************************/
void shmem_float_ixput_( float * target, float * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_ixput wrapper function 
******************************************************/
void shmem_float_ixput__( float * target, float * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_float_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_ixput wrapper function 
******************************************************/
void shmem_double_ixput( double * target, double * source, long * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_ixput wrapper function 
******************************************************/
void SHMEM_DOUBLE_IXPUT( double * target, double * source, long * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_ixput wrapper function 
******************************************************/
void shmem_double_ixput_( double * target, double * source, long * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_ixput wrapper function 
******************************************************/
void shmem_double_ixput__( double * target, double * source, long * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_double_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_ixput wrapper function 
******************************************************/
void shmem_longdouble_ixput( long double * target, long double * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_ixput wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_IXPUT( long double * target, long double * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_ixput wrapper function 
******************************************************/
void shmem_longdouble_ixput_( long double * target, long double * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_ixput wrapper function 
******************************************************/
void shmem_longdouble_ixput__( long double * target, long double * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_ixput wrapper function 
******************************************************/
void shmem_complexf_ixput( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_ixput wrapper function 
******************************************************/
void SHMEM_COMPLEXF_IXPUT( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_ixput wrapper function 
******************************************************/
void shmem_complexf_ixput_( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_ixput wrapper function 
******************************************************/
void shmem_complexf_ixput__( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_ixput wrapper function 
******************************************************/
void shmem_complexd_ixput( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_ixput wrapper function 
******************************************************/
void SHMEM_COMPLEXD_IXPUT( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_ixput wrapper function 
******************************************************/
void shmem_complexd_ixput_( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_ixput wrapper function 
******************************************************/
void shmem_complexd_ixput__( void * target, void * source, int * target_index, int len, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_ixput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_ixput( target, source, target_index, len, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_ixput( target, source, target_index, len, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_and_to_all wrapper function 
******************************************************/
void shmem_short_and_to_all( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_AND_TO_ALL( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all wrapper function 
******************************************************/
void shmem_short_and_to_all_( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_and_to_all wrapper function 
******************************************************/
void shmem_short_and_to_all__( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_and_to_all wrapper function 
******************************************************/
void shmem_int_and_to_all( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all wrapper function 
******************************************************/
void SHMEM_INT_AND_TO_ALL( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all wrapper function 
******************************************************/
void shmem_int_and_to_all_( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_and_to_all wrapper function 
******************************************************/
void shmem_int_and_to_all__( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_and_to_all wrapper function 
******************************************************/
void shmem_long_and_to_all( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all wrapper function 
******************************************************/
void SHMEM_LONG_AND_TO_ALL( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all wrapper function 
******************************************************/
void shmem_long_and_to_all_( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_and_to_all wrapper function 
******************************************************/
void shmem_long_and_to_all__( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_and_to_all wrapper function 
******************************************************/
void shmem_longlong_and_to_all( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_AND_TO_ALL( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all wrapper function 
******************************************************/
void shmem_longlong_and_to_all_( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_and_to_all wrapper function 
******************************************************/
void shmem_longlong_and_to_all__( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_and_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_and_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_or_to_all wrapper function 
******************************************************/
void shmem_short_or_to_all( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_OR_TO_ALL( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all wrapper function 
******************************************************/
void shmem_short_or_to_all_( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_or_to_all wrapper function 
******************************************************/
void shmem_short_or_to_all__( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_or_to_all wrapper function 
******************************************************/
void shmem_int_or_to_all( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all wrapper function 
******************************************************/
void SHMEM_INT_OR_TO_ALL( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all wrapper function 
******************************************************/
void shmem_int_or_to_all_( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_or_to_all wrapper function 
******************************************************/
void shmem_int_or_to_all__( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_or_to_all wrapper function 
******************************************************/
void shmem_long_or_to_all( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all wrapper function 
******************************************************/
void SHMEM_LONG_OR_TO_ALL( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all wrapper function 
******************************************************/
void shmem_long_or_to_all_( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_or_to_all wrapper function 
******************************************************/
void shmem_long_or_to_all__( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_or_to_all wrapper function 
******************************************************/
void shmem_longlong_or_to_all( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_OR_TO_ALL( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all wrapper function 
******************************************************/
void shmem_longlong_or_to_all_( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_or_to_all wrapper function 
******************************************************/
void shmem_longlong_or_to_all__( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_or_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_or_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_xor_to_all wrapper function 
******************************************************/
void shmem_short_xor_to_all( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_XOR_TO_ALL( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all wrapper function 
******************************************************/
void shmem_short_xor_to_all_( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_xor_to_all wrapper function 
******************************************************/
void shmem_short_xor_to_all__( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_xor_to_all wrapper function 
******************************************************/
void shmem_int_xor_to_all( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all wrapper function 
******************************************************/
void SHMEM_INT_XOR_TO_ALL( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all wrapper function 
******************************************************/
void shmem_int_xor_to_all_( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_xor_to_all wrapper function 
******************************************************/
void shmem_int_xor_to_all__( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_xor_to_all wrapper function 
******************************************************/
void shmem_long_xor_to_all( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all wrapper function 
******************************************************/
void SHMEM_LONG_XOR_TO_ALL( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all wrapper function 
******************************************************/
void shmem_long_xor_to_all_( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_xor_to_all wrapper function 
******************************************************/
void shmem_long_xor_to_all__( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_xor_to_all wrapper function 
******************************************************/
void shmem_longlong_xor_to_all( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_XOR_TO_ALL( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all wrapper function 
******************************************************/
void shmem_longlong_xor_to_all_( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_xor_to_all wrapper function 
******************************************************/
void shmem_longlong_xor_to_all__( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_xor_to_all wrapper function 
******************************************************/
void shmem_float_xor_to_all( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_xor_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_XOR_TO_ALL( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_xor_to_all wrapper function 
******************************************************/
void shmem_float_xor_to_all_( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_xor_to_all wrapper function 
******************************************************/
void shmem_float_xor_to_all__( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_xor_to_all wrapper function 
******************************************************/
void shmem_double_xor_to_all( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_xor_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_XOR_TO_ALL( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_xor_to_all wrapper function 
******************************************************/
void shmem_double_xor_to_all_( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_xor_to_all wrapper function 
******************************************************/
void shmem_double_xor_to_all__( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_xor_to_all wrapper function 
******************************************************/
void shmem_longdouble_xor_to_all( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_xor_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_XOR_TO_ALL( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_xor_to_all wrapper function 
******************************************************/
void shmem_longdouble_xor_to_all_( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_xor_to_all wrapper function 
******************************************************/
void shmem_longdouble_xor_to_all__( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_xor_to_all wrapper function 
******************************************************/
void shmem_complexf_xor_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_xor_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXF_XOR_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_xor_to_all wrapper function 
******************************************************/
void shmem_complexf_xor_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_xor_to_all wrapper function 
******************************************************/
void shmem_complexf_xor_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_xor_to_all wrapper function 
******************************************************/
void shmem_complexd_xor_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_xor_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXD_XOR_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_xor_to_all wrapper function 
******************************************************/
void shmem_complexd_xor_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_xor_to_all wrapper function 
******************************************************/
void shmem_complexd_xor_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_xor_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_xor_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_min_to_all wrapper function 
******************************************************/
void shmem_short_min_to_all( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_MIN_TO_ALL( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all wrapper function 
******************************************************/
void shmem_short_min_to_all_( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_min_to_all wrapper function 
******************************************************/
void shmem_short_min_to_all__( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_min_to_all wrapper function 
******************************************************/
void shmem_int_min_to_all( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all wrapper function 
******************************************************/
void SHMEM_INT_MIN_TO_ALL( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all wrapper function 
******************************************************/
void shmem_int_min_to_all_( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_min_to_all wrapper function 
******************************************************/
void shmem_int_min_to_all__( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_min_to_all wrapper function 
******************************************************/
void shmem_long_min_to_all( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all wrapper function 
******************************************************/
void SHMEM_LONG_MIN_TO_ALL( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all wrapper function 
******************************************************/
void shmem_long_min_to_all_( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_min_to_all wrapper function 
******************************************************/
void shmem_long_min_to_all__( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_min_to_all wrapper function 
******************************************************/
void shmem_longlong_min_to_all( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_MIN_TO_ALL( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all wrapper function 
******************************************************/
void shmem_longlong_min_to_all_( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_min_to_all wrapper function 
******************************************************/
void shmem_longlong_min_to_all__( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_min_to_all wrapper function 
******************************************************/
void shmem_float_min_to_all( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_MIN_TO_ALL( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all wrapper function 
******************************************************/
void shmem_float_min_to_all_( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_min_to_all wrapper function 
******************************************************/
void shmem_float_min_to_all__( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_min_to_all wrapper function 
******************************************************/
void shmem_double_min_to_all( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_MIN_TO_ALL( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all wrapper function 
******************************************************/
void shmem_double_min_to_all_( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_min_to_all wrapper function 
******************************************************/
void shmem_double_min_to_all__( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_min_to_all wrapper function 
******************************************************/
void shmem_longdouble_min_to_all( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_MIN_TO_ALL( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all wrapper function 
******************************************************/
void shmem_longdouble_min_to_all_( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_min_to_all wrapper function 
******************************************************/
void shmem_longdouble_min_to_all__( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_min_to_all wrapper function 
******************************************************/
void shmem_complexf_min_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_min_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXF_MIN_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_min_to_all wrapper function 
******************************************************/
void shmem_complexf_min_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_min_to_all wrapper function 
******************************************************/
void shmem_complexf_min_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_min_to_all wrapper function 
******************************************************/
void shmem_complexd_min_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_min_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXD_MIN_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_min_to_all wrapper function 
******************************************************/
void shmem_complexd_min_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_min_to_all wrapper function 
******************************************************/
void shmem_complexd_min_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_min_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_min_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_max_to_all wrapper function 
******************************************************/
void shmem_short_max_to_all( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_MAX_TO_ALL( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all wrapper function 
******************************************************/
void shmem_short_max_to_all_( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_max_to_all wrapper function 
******************************************************/
void shmem_short_max_to_all__( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_max_to_all wrapper function 
******************************************************/
void shmem_int_max_to_all( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all wrapper function 
******************************************************/
void SHMEM_INT_MAX_TO_ALL( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all wrapper function 
******************************************************/
void shmem_int_max_to_all_( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_max_to_all wrapper function 
******************************************************/
void shmem_int_max_to_all__( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_max_to_all wrapper function 
******************************************************/
void shmem_long_max_to_all( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all wrapper function 
******************************************************/
void SHMEM_LONG_MAX_TO_ALL( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all wrapper function 
******************************************************/
void shmem_long_max_to_all_( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_max_to_all wrapper function 
******************************************************/
void shmem_long_max_to_all__( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_max_to_all wrapper function 
******************************************************/
void shmem_longlong_max_to_all( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_MAX_TO_ALL( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all wrapper function 
******************************************************/
void shmem_longlong_max_to_all_( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_max_to_all wrapper function 
******************************************************/
void shmem_longlong_max_to_all__( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_max_to_all wrapper function 
******************************************************/
void shmem_float_max_to_all( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_MAX_TO_ALL( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all wrapper function 
******************************************************/
void shmem_float_max_to_all_( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_max_to_all wrapper function 
******************************************************/
void shmem_float_max_to_all__( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_max_to_all wrapper function 
******************************************************/
void shmem_double_max_to_all( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_MAX_TO_ALL( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all wrapper function 
******************************************************/
void shmem_double_max_to_all_( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_max_to_all wrapper function 
******************************************************/
void shmem_double_max_to_all__( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_max_to_all wrapper function 
******************************************************/
void shmem_longdouble_max_to_all( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_MAX_TO_ALL( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all wrapper function 
******************************************************/
void shmem_longdouble_max_to_all_( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_max_to_all wrapper function 
******************************************************/
void shmem_longdouble_max_to_all__( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_max_to_all wrapper function 
******************************************************/
void shmem_complexf_max_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_max_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXF_MAX_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_max_to_all wrapper function 
******************************************************/
void shmem_complexf_max_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_max_to_all wrapper function 
******************************************************/
void shmem_complexf_max_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_max_to_all wrapper function 
******************************************************/
void shmem_complexd_max_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_max_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXD_MAX_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_max_to_all wrapper function 
******************************************************/
void shmem_complexd_max_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_max_to_all wrapper function 
******************************************************/
void shmem_complexd_max_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_max_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_max_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_prod_to_all wrapper function 
******************************************************/
void shmem_short_prod_to_all( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_PROD_TO_ALL( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all wrapper function 
******************************************************/
void shmem_short_prod_to_all_( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_prod_to_all wrapper function 
******************************************************/
void shmem_short_prod_to_all__( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_prod_to_all wrapper function 
******************************************************/
void shmem_int_prod_to_all( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all wrapper function 
******************************************************/
void SHMEM_INT_PROD_TO_ALL( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all wrapper function 
******************************************************/
void shmem_int_prod_to_all_( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_prod_to_all wrapper function 
******************************************************/
void shmem_int_prod_to_all__( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_prod_to_all wrapper function 
******************************************************/
void shmem_long_prod_to_all( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all wrapper function 
******************************************************/
void SHMEM_LONG_PROD_TO_ALL( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all wrapper function 
******************************************************/
void shmem_long_prod_to_all_( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_prod_to_all wrapper function 
******************************************************/
void shmem_long_prod_to_all__( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_prod_to_all wrapper function 
******************************************************/
void shmem_longlong_prod_to_all( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_PROD_TO_ALL( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all wrapper function 
******************************************************/
void shmem_longlong_prod_to_all_( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_prod_to_all wrapper function 
******************************************************/
void shmem_longlong_prod_to_all__( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_prod_to_all wrapper function 
******************************************************/
void shmem_float_prod_to_all( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_PROD_TO_ALL( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all wrapper function 
******************************************************/
void shmem_float_prod_to_all_( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_prod_to_all wrapper function 
******************************************************/
void shmem_float_prod_to_all__( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_prod_to_all wrapper function 
******************************************************/
void shmem_double_prod_to_all( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_PROD_TO_ALL( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all wrapper function 
******************************************************/
void shmem_double_prod_to_all_( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_prod_to_all wrapper function 
******************************************************/
void shmem_double_prod_to_all__( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_prod_to_all wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_PROD_TO_ALL( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all_( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_prod_to_all wrapper function 
******************************************************/
void shmem_longdouble_prod_to_all__( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_prod_to_all wrapper function 
******************************************************/
void shmem_complexf_prod_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXF_PROD_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all wrapper function 
******************************************************/
void shmem_complexf_prod_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_prod_to_all wrapper function 
******************************************************/
void shmem_complexf_prod_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_prod_to_all wrapper function 
******************************************************/
void shmem_complexd_prod_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXD_PROD_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all wrapper function 
******************************************************/
void shmem_complexd_prod_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_prod_to_all wrapper function 
******************************************************/
void shmem_complexd_prod_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_prod_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_prod_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_sum_to_all wrapper function 
******************************************************/
void shmem_short_sum_to_all( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all wrapper function 
******************************************************/
void SHMEM_SHORT_SUM_TO_ALL( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all wrapper function 
******************************************************/
void shmem_short_sum_to_all_( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_sum_to_all wrapper function 
******************************************************/
void shmem_short_sum_to_all__( short * target, short * source, int nreduce, int PE_start, int logPE_stride, int PE_size, short * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_sum_to_all wrapper function 
******************************************************/
void shmem_int_sum_to_all( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all wrapper function 
******************************************************/
void SHMEM_INT_SUM_TO_ALL( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all wrapper function 
******************************************************/
void shmem_int_sum_to_all_( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_sum_to_all wrapper function 
******************************************************/
void shmem_int_sum_to_all__( int * target, int * source, int nreduce, int PE_start, int logPE_stride, int PE_size, int * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_sum_to_all wrapper function 
******************************************************/
void shmem_long_sum_to_all( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all wrapper function 
******************************************************/
void SHMEM_LONG_SUM_TO_ALL( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all wrapper function 
******************************************************/
void shmem_long_sum_to_all_( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_sum_to_all wrapper function 
******************************************************/
void shmem_long_sum_to_all__( long * target, long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_sum_to_all wrapper function 
******************************************************/
void shmem_longlong_sum_to_all( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all wrapper function 
******************************************************/
void SHMEM_LONGLONG_SUM_TO_ALL( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all wrapper function 
******************************************************/
void shmem_longlong_sum_to_all_( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_sum_to_all wrapper function 
******************************************************/
void shmem_longlong_sum_to_all__( long long * target, long long * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_sum_to_all wrapper function 
******************************************************/
void shmem_float_sum_to_all( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all wrapper function 
******************************************************/
void SHMEM_FLOAT_SUM_TO_ALL( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all wrapper function 
******************************************************/
void shmem_float_sum_to_all_( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_sum_to_all wrapper function 
******************************************************/
void shmem_float_sum_to_all__( float * target, float * source, int nreduce, int PE_start, int logPE_stride, int PE_size, float * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_sum_to_all wrapper function 
******************************************************/
void shmem_double_sum_to_all( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all wrapper function 
******************************************************/
void SHMEM_DOUBLE_SUM_TO_ALL( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all wrapper function 
******************************************************/
void shmem_double_sum_to_all_( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_sum_to_all wrapper function 
******************************************************/
void shmem_double_sum_to_all__( double * target, double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_sum_to_all wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_SUM_TO_ALL( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all_( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_sum_to_all wrapper function 
******************************************************/
void shmem_longdouble_sum_to_all__( long double * target, long double * source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_sum_to_all wrapper function 
******************************************************/
void shmem_complexf_sum_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXF_SUM_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all wrapper function 
******************************************************/
void shmem_complexf_sum_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_sum_to_all wrapper function 
******************************************************/
void shmem_complexf_sum_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_sum_to_all wrapper function 
******************************************************/
void shmem_complexd_sum_to_all( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all wrapper function 
******************************************************/
void SHMEM_COMPLEXD_SUM_TO_ALL( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all wrapper function 
******************************************************/
void shmem_complexd_sum_to_all_( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_sum_to_all wrapper function 
******************************************************/
void shmem_complexd_sum_to_all__( void * target, void * source, int nreduce, int PE_start, int logPE_stride, int PE_size, void * pWrk, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_sum_to_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_sum_to_all( target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_alltoall wrapper function 
******************************************************/
void shmem_short_alltoall( short * target, short * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_alltoall wrapper function 
******************************************************/
void SHMEM_SHORT_ALLTOALL( short * target, short * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_alltoall wrapper function 
******************************************************/
void shmem_short_alltoall_( short * target, short * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_alltoall wrapper function 
******************************************************/
void shmem_short_alltoall__( short * target, short * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_short_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_alltoall wrapper function 
******************************************************/
void shmem_int_alltoall( int * target, int * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_alltoall wrapper function 
******************************************************/
void SHMEM_INT_ALLTOALL( int * target, int * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_alltoall wrapper function 
******************************************************/
void shmem_int_alltoall_( int * target, int * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_alltoall wrapper function 
******************************************************/
void shmem_int_alltoall__( int * target, int * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_int_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_alltoall wrapper function 
******************************************************/
void shmem_long_alltoall( long * target, long * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_alltoall wrapper function 
******************************************************/
void SHMEM_LONG_ALLTOALL( long * target, long * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_alltoall wrapper function 
******************************************************/
void shmem_long_alltoall_( long * target, long * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_alltoall wrapper function 
******************************************************/
void shmem_long_alltoall__( long * target, long * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_long_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_alltoall wrapper function 
******************************************************/
void shmem_longlong_alltoall( long long * target, long long * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_alltoall wrapper function 
******************************************************/
void SHMEM_LONGLONG_ALLTOALL( long long * target, long long * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_alltoall wrapper function 
******************************************************/
void shmem_longlong_alltoall_( long long * target, long long * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_alltoall wrapper function 
******************************************************/
void shmem_longlong_alltoall__( long long * target, long long * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_alltoall wrapper function 
******************************************************/
void shmem_float_alltoall( float * target, float * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_alltoall wrapper function 
******************************************************/
void SHMEM_FLOAT_ALLTOALL( float * target, float * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_alltoall wrapper function 
******************************************************/
void shmem_float_alltoall_( float * target, float * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_float_alltoall wrapper function 
******************************************************/
void shmem_float_alltoall__( float * target, float * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_float_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_float_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_float_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_alltoall wrapper function 
******************************************************/
void shmem_double_alltoall( double * target, double * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_alltoall wrapper function 
******************************************************/
void SHMEM_DOUBLE_ALLTOALL( double * target, double * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_alltoall wrapper function 
******************************************************/
void shmem_double_alltoall_( double * target, double * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_double_alltoall wrapper function 
******************************************************/
void shmem_double_alltoall__( double * target, double * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_double_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_double_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_double_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longdouble_alltoall wrapper function 
******************************************************/
void shmem_longdouble_alltoall( long double * target, long double * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_alltoall wrapper function 
******************************************************/
void SHMEM_LONGDOUBLE_ALLTOALL( long double * target, long double * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_alltoall wrapper function 
******************************************************/
void shmem_longdouble_alltoall_( long double * target, long double * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longdouble_alltoall wrapper function 
******************************************************/
void shmem_longdouble_alltoall__( long double * target, long double * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_longdouble_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longdouble_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longdouble_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexf_alltoall wrapper function 
******************************************************/
void shmem_complexf_alltoall( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_alltoall wrapper function 
******************************************************/
void SHMEM_COMPLEXF_ALLTOALL( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_alltoall wrapper function 
******************************************************/
void shmem_complexf_alltoall_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexf_alltoall wrapper function 
******************************************************/
void shmem_complexf_alltoall__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexf_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexf_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexf_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_complexd_alltoall wrapper function 
******************************************************/
void shmem_complexd_alltoall( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_alltoall wrapper function 
******************************************************/
void SHMEM_COMPLEXD_ALLTOALL( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_alltoall wrapper function 
******************************************************/
void shmem_complexd_alltoall_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_complexd_alltoall wrapper function 
******************************************************/
void shmem_complexd_alltoall__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_complexd_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_complexd_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_complexd_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_alltoall wrapper function 
******************************************************/
void shmem_alltoall( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall wrapper function 
******************************************************/
void SHMEM_ALLTOALL( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall wrapper function 
******************************************************/
void shmem_alltoall_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall wrapper function 
******************************************************/
void shmem_alltoall__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_alltoall32 wrapper function 
******************************************************/
void shmem_alltoall32( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall32 wrapper function 
******************************************************/
void SHMEM_ALLTOALL32( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall32 wrapper function 
******************************************************/
void shmem_alltoall32_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall32 wrapper function 
******************************************************/
void shmem_alltoall32__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall32()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall32( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_alltoall64 wrapper function 
******************************************************/
void shmem_alltoall64( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall64 wrapper function 
******************************************************/
void SHMEM_ALLTOALL64( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall64 wrapper function 
******************************************************/
void shmem_alltoall64_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall64 wrapper function 
******************************************************/
void shmem_alltoall64__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall64()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall64( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_alltoall128 wrapper function 
******************************************************/
void shmem_alltoall128( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall128 wrapper function 
******************************************************/
void SHMEM_ALLTOALL128( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall128 wrapper function 
******************************************************/
void shmem_alltoall128_( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_alltoall128 wrapper function 
******************************************************/
void shmem_alltoall128__( void * target, void * source, int nlong, int PE_start, int logPE_stride, int PE_size, long * pSync)
{

  TAU_PROFILE_TIMER(t, "shmem_alltoall128()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_alltoall128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_alltoall128( target, source, nlong, PE_start, logPE_stride, PE_size, pSync) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_cswap wrapper function 
******************************************************/
short shmem_short_cswap( short * target, short * val, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap wrapper function 
******************************************************/
short SHMEM_SHORT_CSWAP( short * target, short * val, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap wrapper function 
******************************************************/
short shmem_short_cswap_( short * target, short * val, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_cswap wrapper function 
******************************************************/
short shmem_short_cswap__( short * target, short * val, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_cswap wrapper function 
******************************************************/
int shmem_int_cswap( int * target, int * val, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap wrapper function 
******************************************************/
int SHMEM_INT_CSWAP( int * target, int * val, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap wrapper function 
******************************************************/
int shmem_int_cswap_( int * target, int * val, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_cswap wrapper function 
******************************************************/
int shmem_int_cswap__( int * target, int * val, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_cswap wrapper function 
******************************************************/
long shmem_long_cswap( long * target, long * val, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap wrapper function 
******************************************************/
long SHMEM_LONG_CSWAP( long * target, long * val, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap wrapper function 
******************************************************/
long shmem_long_cswap_( long * target, long * val, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_cswap wrapper function 
******************************************************/
long shmem_long_cswap__( long * target, long * val, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_cswap wrapper function 
******************************************************/
long long shmem_longlong_cswap( long long * target, long long * val, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap wrapper function 
******************************************************/
long long SHMEM_LONGLONG_CSWAP( long long * target, long long * val, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap wrapper function 
******************************************************/
long long shmem_longlong_cswap_( long long * target, long long * val, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_cswap wrapper function 
******************************************************/
long long shmem_longlong_cswap__( long long * target, long long * val, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_cswap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_cswap( target, val, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_cswap( target, val, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_add wrapper function 
******************************************************/
void shmem_int_add( int * target, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add wrapper function 
******************************************************/
void SHMEM_INT_ADD( int * target, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add wrapper function 
******************************************************/
void shmem_int_add_( int * target, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_add wrapper function 
******************************************************/
void shmem_int_add__( int * target, int value, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_add( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_add( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_fadd wrapper function 
******************************************************/
short shmem_short_fadd( short * target, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd wrapper function 
******************************************************/
short SHMEM_SHORT_FADD( short * target, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd wrapper function 
******************************************************/
short shmem_short_fadd_( short * target, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_fadd wrapper function 
******************************************************/
short shmem_short_fadd__( short * target, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_fadd wrapper function 
******************************************************/
int shmem_int_fadd( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd wrapper function 
******************************************************/
int SHMEM_INT_FADD( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd wrapper function 
******************************************************/
int shmem_int_fadd_( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_fadd wrapper function 
******************************************************/
int shmem_int_fadd__( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_fadd()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_fadd( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_fadd( target, value, pe) ; 
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
***      shmem_short_inc wrapper function 
******************************************************/
void shmem_short_inc( short * target, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc wrapper function 
******************************************************/
void SHMEM_SHORT_INC( short * target, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc wrapper function 
******************************************************/
void shmem_short_inc_( short * target, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_inc wrapper function 
******************************************************/
void shmem_short_inc__( short * target, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_short_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_inc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_inc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_inc wrapper function 
******************************************************/
void shmem_int_inc( int * target, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc wrapper function 
******************************************************/
void SHMEM_INT_INC( int * target, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc wrapper function 
******************************************************/
void shmem_int_inc_( int * target, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_inc wrapper function 
******************************************************/
void shmem_int_inc__( int * target, int pe)
{

  TAU_PROFILE_TIMER(t, "shmem_int_inc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_inc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_inc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_swap wrapper function 
******************************************************/
int shmem_swap( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap wrapper function 
******************************************************/
int SHMEM_SWAP( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap wrapper function 
******************************************************/
int shmem_swap_( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_swap wrapper function 
******************************************************/
int shmem_swap__( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_swap wrapper function 
******************************************************/
short shmem_short_swap( short * target, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap wrapper function 
******************************************************/
short SHMEM_SHORT_SWAP( short * target, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap wrapper function 
******************************************************/
short shmem_short_swap_( short * target, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_swap wrapper function 
******************************************************/
short shmem_short_swap__( short * target, short value, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_swap wrapper function 
******************************************************/
int shmem_int_swap( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap wrapper function 
******************************************************/
int SHMEM_INT_SWAP( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap wrapper function 
******************************************************/
int shmem_int_swap_( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_swap wrapper function 
******************************************************/
int shmem_int_swap__( int * target, int value, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_swap wrapper function 
******************************************************/
long shmem_long_swap( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap wrapper function 
******************************************************/
long SHMEM_LONG_SWAP( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap wrapper function 
******************************************************/
long shmem_long_swap_( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_long_swap wrapper function 
******************************************************/
long shmem_long_swap__( long * target, long value, int pe)
{
  long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_long_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_long_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_long_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_float_swap wrapper function 
******************************************************/
float shmem_float_swap( float * target, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap wrapper function 
******************************************************/
float SHMEM_FLOAT_SWAP( float * target, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap wrapper function 
******************************************************/
float shmem_float_swap_( float * target, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_float_swap wrapper function 
******************************************************/
float shmem_float_swap__( float * target, float value, int pe)
{
  float retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_float_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_float_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_float_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_double_swap wrapper function 
******************************************************/
double shmem_double_swap( double * target, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap wrapper function 
******************************************************/
double SHMEM_DOUBLE_SWAP( double * target, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap wrapper function 
******************************************************/
double shmem_double_swap_( double * target, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_double_swap wrapper function 
******************************************************/
double shmem_double_swap__( double * target, double value, int pe)
{
  double retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_double_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_double_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_double_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_swap wrapper function 
******************************************************/
long long shmem_longlong_swap( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap wrapper function 
******************************************************/
long long SHMEM_LONGLONG_SWAP( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap wrapper function 
******************************************************/
long long shmem_longlong_swap_( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_longlong_swap wrapper function 
******************************************************/
long long shmem_longlong_swap__( long long * target, long long value, int pe)
{
  long long retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_longlong_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_longlong_swap( target, value, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_longlong_swap( target, value, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_finc wrapper function 
******************************************************/
short shmem_short_finc( short * target, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc wrapper function 
******************************************************/
short SHMEM_SHORT_FINC( short * target, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc wrapper function 
******************************************************/
short shmem_short_finc_( short * target, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_short_finc wrapper function 
******************************************************/
short shmem_short_finc__( short * target, int pe)
{
  short retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_short_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_short_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_short_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_finc wrapper function 
******************************************************/
int shmem_int_finc( int * target, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc wrapper function 
******************************************************/
int SHMEM_INT_FINC( int * target, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc wrapper function 
******************************************************/
int shmem_int_finc_( int * target, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc( target, pe) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      shmem_int_finc wrapper function 
******************************************************/
int shmem_int_finc__( int * target, int pe)
{
  int retvalue; 

  TAU_PROFILE_TIMER(t, "shmem_int_finc()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  retvalue = __p__shmem_int_finc( target, pe) ; 
#else /* !TAU_P_SHMEM */ 
  retvalue = pshmem_int_finc( target, pe) ; 
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
***      shmem_clear_lock wrapper function 
******************************************************/
void shmem_clear_lock( int * lock)
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
void SHMEM_CLEAR_LOCK( int * lock)
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
void shmem_clear_lock_( int * lock)
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
void shmem_clear_lock__( int * lock)
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
***      shmem_set_lock wrapper function 
******************************************************/
void shmem_set_lock( int * lock)
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
void SHMEM_SET_LOCK( int * lock)
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
void shmem_set_lock_( int * lock)
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
void shmem_set_lock__( int * lock)
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
***      shmem_test_lock wrapper function 
******************************************************/
int shmem_test_lock( int * lock)
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
int SHMEM_TEST_LOCK( int * lock)
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
int shmem_test_lock_( int * lock)
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
int shmem_test_lock__( int * lock)
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
***      shmem_wait wrapper function 
******************************************************/
void shmem_wait( int * ivar, int cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait wrapper function 
******************************************************/
void SHMEM_WAIT( int * ivar, int cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait wrapper function 
******************************************************/
void shmem_wait_( int * ivar, int cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait wrapper function 
******************************************************/
void shmem_wait__( int * ivar, int cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_wait wrapper function 
******************************************************/
void shmem_short_wait( short * ivar, short cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait wrapper function 
******************************************************/
void SHMEM_SHORT_WAIT( short * ivar, short cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait wrapper function 
******************************************************/
void shmem_short_wait_( short * ivar, short cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait wrapper function 
******************************************************/
void shmem_short_wait__( short * ivar, short cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_wait wrapper function 
******************************************************/
void shmem_int_wait( int * ivar, int cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait wrapper function 
******************************************************/
void SHMEM_INT_WAIT( int * ivar, int cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait wrapper function 
******************************************************/
void shmem_int_wait_( int * ivar, int cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait wrapper function 
******************************************************/
void shmem_int_wait__( int * ivar, int cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_wait wrapper function 
******************************************************/
void shmem_long_wait( long * ivar, long cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait wrapper function 
******************************************************/
void SHMEM_LONG_WAIT( long * ivar, long cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait wrapper function 
******************************************************/
void shmem_long_wait_( long * ivar, long cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait wrapper function 
******************************************************/
void shmem_long_wait__( long * ivar, long cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_wait wrapper function 
******************************************************/
void shmem_longlong_wait( long long * ivar, long long cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait wrapper function 
******************************************************/
void SHMEM_LONGLONG_WAIT( long long * ivar, long long cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait wrapper function 
******************************************************/
void shmem_longlong_wait_( long long * ivar, long long cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait wrapper function 
******************************************************/
void shmem_longlong_wait__( long long * ivar, long long cmp_value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait( ivar, cmp_value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait( ivar, cmp_value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_wait_until wrapper function 
******************************************************/
void shmem_wait_until( int * ivar, int cmp, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_until wrapper function 
******************************************************/
void SHMEM_WAIT_UNTIL( int * ivar, int cmp, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_until wrapper function 
******************************************************/
void shmem_wait_until_( int * ivar, int cmp, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_wait_until wrapper function 
******************************************************/
void shmem_wait_until__( int * ivar, int cmp, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_short_wait_until wrapper function 
******************************************************/
void shmem_short_wait_until( short * ivar, int cmp, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait_until wrapper function 
******************************************************/
void SHMEM_SHORT_WAIT_UNTIL( short * ivar, int cmp, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait_until wrapper function 
******************************************************/
void shmem_short_wait_until_( short * ivar, int cmp, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_short_wait_until wrapper function 
******************************************************/
void shmem_short_wait_until__( short * ivar, int cmp, short value)
{

  TAU_PROFILE_TIMER(t, "shmem_short_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_short_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_short_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_int_wait_until wrapper function 
******************************************************/
void shmem_int_wait_until( int * ivar, int cmp, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait_until wrapper function 
******************************************************/
void SHMEM_INT_WAIT_UNTIL( int * ivar, int cmp, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait_until wrapper function 
******************************************************/
void shmem_int_wait_until_( int * ivar, int cmp, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_int_wait_until wrapper function 
******************************************************/
void shmem_int_wait_until__( int * ivar, int cmp, int value)
{

  TAU_PROFILE_TIMER(t, "shmem_int_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_int_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_int_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_long_wait_until wrapper function 
******************************************************/
void shmem_long_wait_until( long * ivar, int cmp, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait_until wrapper function 
******************************************************/
void SHMEM_LONG_WAIT_UNTIL( long * ivar, int cmp, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait_until wrapper function 
******************************************************/
void shmem_long_wait_until_( long * ivar, int cmp, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_long_wait_until wrapper function 
******************************************************/
void shmem_long_wait_until__( long * ivar, int cmp, long value)
{

  TAU_PROFILE_TIMER(t, "shmem_long_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_long_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_long_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      shmem_longlong_wait_until wrapper function 
******************************************************/
void shmem_longlong_wait_until( long long * ivar, int cmp, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait_until wrapper function 
******************************************************/
void SHMEM_LONGLONG_WAIT_UNTIL( long long * ivar, int cmp, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait_until wrapper function 
******************************************************/
void shmem_longlong_wait_until_( long long * ivar, int cmp, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

/******************************************************
***      shmem_longlong_wait_until wrapper function 
******************************************************/
void shmem_longlong_wait_until__( long long * ivar, int cmp, long long value)
{

  TAU_PROFILE_TIMER(t, "shmem_longlong_wait_until()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
#ifdef TAU_P_SHMEM 
  __p__shmem_longlong_wait_until( ivar, cmp, value) ; 
#else /* !TAU_P_SHMEM */ 
  pshmem_longlong_wait_until( ivar, cmp, value) ; 
#endif /* TAU_P_SHMEM */ 
  TAU_PROFILE_STOP(t); 
  return ; 
}

