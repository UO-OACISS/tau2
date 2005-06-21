#include "gpshmem.h"
#include <stdarg.h>
#include "gps_pshmem.h"
#include "TAU.h"
/******************************************************/
/**  PGPSHMEM Wrapper Library, (C) 2005 Adam Leko
     HCS Lab, University of Florida                   */
/******************************************************/


/******************************************************
*** GPSHMEM C init wrapper function (init/gp_init.c)
******************************************************/
int 
GPSHMEM_INIT_C (int* argc, char*** argv)
{
  int retval;
  TAU_PROFILE_TIMER(t, "GPSHMEM_INIT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = PGPSHMEM_INIT_C(argc, argv);
  TAU_PROFILE_SET_NODE(GPMYPE_C());
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHMEM Fortran init wrapper function (init/gp_init.c)
******************************************************/
int 
GPSHMEM_INIT_CORE_F (int* ptr_num, ...)
{
  va_list ap;
  int retval;
  /* Note that this function duplicates a wrapper function
     in gp_init.c, but is necessary due to the wierdness of
     trying to portably copy the ... above (from stdarg.h) */
  TAU_PROFILE_TIMER(t, "GPSHMEM_INIT_CORE_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  gp_init();
  va_start(ap, ptr_num);
  retval = gp_init_ma77(*ptr_num, ap);
  va_end(ap);
  TAU_PROFILE_SET_NODE(GPMYPE_C());
  TAU_PROFILE_STOP(t);
  return retval;
}



/******************************************************
*** GPBARRIER_C wrapper function 
*** (from barrier/gps_barrier.c)
******************************************************/
void 
GPBARRIER_C (void)
{
  TAU_PROFILE_TIMER(t, "GPBARRIER_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPBARRIER_C();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BARRIER_ALL_C wrapper function 
*** (from barrier/gps_barrier.c)
******************************************************/
void 
GPSHMEM_BARRIER_ALL_C (void)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BARRIER_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BARRIER_ALL_C();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BARRIER__ST_C wrapper function 
*** (from barrier/gps_barrier.c)
******************************************************/
void 
GPSHMEM_BARRIER__ST_C (int PE_start, int PE_stride, int PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BARRIER__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BARRIER__ST_C(PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BARRIER_C wrapper function 
*** (from barrier/gps_barrier.c)
******************************************************/
void 
GPSHMEM_BARRIER_C (int PE_start, int logPE_stride, int PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BARRIER_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BARRIER_C(PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPBARRIER_F wrapper function 
*** (from barrier/gps_barrier.c)
******************************************************/
void 
GPBARRIER_F (void)
{
  TAU_PROFILE_TIMER(t, "GPBARRIER_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPBARRIER_F();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BARRIER_ALL_F wrapper function 
*** (from barrier/gps_barrier.c)
******************************************************/
void 
GPSHMEM_BARRIER_ALL_F (void)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BARRIER_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BARRIER_ALL_F();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BARRIER__ST_F wrapper function 
*** (from barrier/gps_barrier.c)
******************************************************/
void 
GPSHMEM_BARRIER__ST_F (int* PE_start, int* PE_stride, int* PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BARRIER__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BARRIER__ST_F(PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BARRIER_F wrapper function 
*** (from barrier/gps_barrier.c)
******************************************************/
void 
GPSHMEM_BARRIER_F (int* PE_start, int* logPE_stride, int* PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BARRIER_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BARRIER_F(PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BROADCAST__ST_C wrapper function 
*** (from broadcast/gps_bcast.c)
******************************************************/
void 
GPSHMEM_BROADCAST__ST_C (long* target, long* source, int nlong,
			int PE_root, int PE_start, int PE_stride,
			int PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BROADCAST__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BROADCAST__ST_C(target, source, nlong, PE_root, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BROADCAST32__ST_C wrapper function 
*** (from broadcast/gps_bcast.c)
******************************************************/
void 
GPSHMEM_BROADCAST32__ST_C (short* target, short* source, int nlong,
			int PE_root, int PE_start, int PE_stride,
		      	int PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BROADCAST32__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BROADCAST32__ST_C(target, source, nlong, PE_root, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BROADCAST_C wrapper function 
*** (from broadcast/gps_bcast.c)
******************************************************/
void 
GPSHMEM_BROADCAST_C (long* target, long* source, int nlong,
		    int PE_root, int PE_start, int logPE_stride,
		    int PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BROADCAST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BROADCAST_C(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BROADCAST32_C wrapper function 
*** (from broadcast/gps_bcast.c)
******************************************************/
void 
GPSHMEM_BROADCAST32_C (short* target, short* source, int nlong,
		      int PE_root, int PE_start, int logPE_stride,
		      int PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BROADCAST32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BROADCAST32_C(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BROADCAST__ST_F wrapper function 
*** (from broadcast/gps_bcast.c)
******************************************************/
void 
GPSHMEM_BROADCAST__ST_F (long* target, long* source, int* nlong,
			int* PE_root, int* PE_start, int* PE_stride,
			int* PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BROADCAST__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BROADCAST__ST_F(target, source, nlong, PE_root, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BROADCAST32__ST_F wrapper function 
*** (from broadcast/gps_bcast.c)
******************************************************/
void 
GPSHMEM_BROADCAST32__ST_F (short* target, short* source, int* nlong,
			int* PE_root, int* PE_start, int* PE_stride,
			int* PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BROADCAST32__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BROADCAST32__ST_F(target, source, nlong, PE_root, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BROADCAST_F wrapper function 
*** (from broadcast/gps_bcast.c)
******************************************************/
void 
GPSHMEM_BROADCAST_F (long* target, long* source, int* nlong,
		    int* PE_root, int* PE_start, int* logPE_stride,
		    int* PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BROADCAST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BROADCAST_F(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_BROADCAST32_F wrapper function 
*** (from broadcast/gps_bcast.c)
******************************************************/
void 
GPSHMEM_BROADCAST32_F (short* target, short* source, int* nlong,
		      int* PE_root, int* PE_start, int* logPE_stride,
		      int* PE_size, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_BROADCAST32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_BROADCAST32_F(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_COLLECT__ST_C wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_COLLECT__ST_C (long* target, long* source, int nwords,
		int PE_start, int PE_stride, int PE_size,
		long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_COLLECT__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_COLLECT__ST_C(target, source, nwords, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_COLLECT32__ST_C wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_COLLECT32__ST_C (short* target, short* source, int nwords,
		    int PE_start, int PE_stride, int PE_size,
		    long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_COLLECT32__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_COLLECT32__ST_C(target, source, nwords, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FCOLLECT__ST_C wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_FCOLLECT__ST_C (long* target, long* source, int nwords,
		   int PE_start, int PE_stride, int PE_size,
		   long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FCOLLECT__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FCOLLECT__ST_C(target, source, nwords, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FCOLLECT32__ST_C wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_FCOLLECT32__ST_C (short* target, short* source, int nwords,
		     int PE_start, int PE_stride, int PE_size,
		     long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FCOLLECT32__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FCOLLECT32__ST_C(target, source, nwords, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_COLLECT_C wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_COLLECT_C (long* target, long* source, int nwords,
		int PE_start, int logPE_stride, int PE_size,
		long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_COLLECT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_COLLECT_C(target, source, nwords, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_COLLECT32_C wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_COLLECT32_C (short* target, short* source, int nwords,
		    int PE_start, int logPE_stride, int PE_size,
		    long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_COLLECT32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_COLLECT32_C(target, source, nwords, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FCOLLECT_C wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_FCOLLECT_C (long* target, long* source, int nwords,
		   int PE_start, int logPE_stride, int PE_size,
		   long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FCOLLECT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FCOLLECT_C(target, source, nwords, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FCOLLECT32_C wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_FCOLLECT32_C (short* target, short* source, int nwords,
		     int PE_start, int logPE_stride, int PE_size,
		     long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FCOLLECT32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FCOLLECT32_C(target, source, nwords, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_COLLECT__ST_F wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_COLLECT__ST_F (long* target, long* source, int* nwords,
		  int* PE_start, int* PE_stride, int* PE_size,
		  long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_COLLECT__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_COLLECT__ST_F(target, source, nwords, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_COLLECT32__ST_F wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_COLLECT32__ST_F (short* target, short* source, int* nwords,
		    int* PE_start, int* PE_stride, int* PE_size,
		    long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_COLLECT32__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_COLLECT32__ST_F(target, source, nwords, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FCOLLECT__ST_F wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_FCOLLECT__ST_F (long* target, long* source, int* nwords,
		   int* PE_start, int* PE_stride, int* PE_size,
		   long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FCOLLECT__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FCOLLECT__ST_F(target, source, nwords, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FCOLLECT32__ST_F wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_FCOLLECT32__ST_F (short* target, short* source, int* nwords,
		     int* PE_start, int* PE_stride, int* PE_size,
		     long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FCOLLECT32__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FCOLLECT32__ST_F(target, source, nwords, PE_start, PE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_COLLECT_F wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_COLLECT_F (long* target, long* source, int* nwords,
		  int* PE_start, int* logPE_stride, int* PE_size,
		  long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_COLLECT_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_COLLECT_F(target, source, nwords, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_COLLECT32_F wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_COLLECT32_F (short* target, short* source, int* nwords,
		    int* PE_start, int* logPE_stride, int* PE_size,
		    long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_COLLECT32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_COLLECT32_F(target, source, nwords, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FCOLLECT_F wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_FCOLLECT_F (long* target, long* source, int* nwords,
		   int* PE_start, int* logPE_stride, int* PE_size,
		   long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FCOLLECT_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FCOLLECT_F(target, source, nwords, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FCOLLECT32_F wrapper function 
*** (from collect/gps_collect.c)
******************************************************/
void 
GPSHMEM_FCOLLECT32_F (short* target, short* source, int* nwords,
		     int* PE_start, int* logPE_stride, int* PE_size,
		     long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FCOLLECT32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FCOLLECT32_F(target, source, nwords, PE_start, logPE_stride, PE_size, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_GETMEM_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_GETMEM_C (void* target, void* source, int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_GETMEM_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_GETMEM_C(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_GET_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_GET_C (long* target, long* source, int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_GET_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_GET_C(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_GET32_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_GET32_C (short* target, short* source, int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_GET32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_GET32_C(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_PUTMEM_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_PUTMEM_C (void* target, void* source, int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_PUTMEM_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_PUTMEM_C(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_PUT_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_PUT_C (long* target, long* source, int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_PUT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_PUT_C(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_PUT32_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_PUT32_C (short* target, short* source, int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_PUT32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_PUT32_C(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IGET_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IGET_C (long* target, long* source, int target_inc,
	       int source_inc, int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IGET_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IGET_C(target, source, target_inc, source_inc, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IGET32_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IGET32_C (short* target, short* source, int target_inc,
		 int source_inc,int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IGET32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IGET32_C(target, source, target_inc, source_inc, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IPUT_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IPUT_C (long* target, long* source, int target_inc,
	       int source_inc, int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IPUT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IPUT_C(target, source, target_inc, source_inc, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IPUT32_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IPUT32_C (short* target, short* source, int target_inc,
		 int source_inc,int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IPUT32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IPUT32_C(target, source, target_inc, source_inc, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IXGET_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IXGET_C (long* target, long* source, long* source_index,
		int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IXGET_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IXGET_C(target, source, source_index, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IXGET32_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IXGET32_C (short* target, short* source, short* source_index,
		  int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IXGET32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IXGET32_C(target, source, source_index, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IXPUT_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IXPUT_C (long* target, long* source, long* target_index,
		int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IXPUT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IXPUT_C(target, source, target_index, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IXPUT32_C wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IXPUT32_C (short* target, short* source, short* target_index,
		  int nlong, int pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IXPUT32_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IXPUT32_C(target, source, target_index, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_GETMEM_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_GETMEM_F (void* target, void* source, int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_GETMEM_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_GETMEM_F(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_GET_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_GET_F (long* target, long* source, int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_GET_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_GET_F(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_GET32_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_GET32_F (short* target, short* source, int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_GET32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_GET32_F(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_PUT_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_PUT_F (long* target, long* source, int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_PUT_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_PUT_F(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_PUT32_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_PUT32_F (short* target, short* source, int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_PUT32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_PUT32_F(target, source, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IGET_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IGET_F (long* target, long* source, int* target_inc,
	       int* source_inc, int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IGET_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IGET_F(target, source, target_inc, source_inc, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IGET32_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IGET32_F (short* target, short* source, int* target_inc,
		 int* source_inc,int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IGET32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IGET32_F(target, source, target_inc, source_inc, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IPUT_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IPUT_F (long* target, long* source, int* target_inc,
	       int* source_inc, int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IPUT_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IPUT_F(target, source, target_inc, source_inc, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IPUT32_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IPUT32_F (short* target, short* source, int* target_inc,
		 int* source_inc,int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IPUT32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IPUT32_F(target, source, target_inc, source_inc, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IXGET_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IXGET_F (long* target, long* source, long* source_index,
		int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IXGET_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IXGET_F(target, source, source_index, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IXGET32_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IXGET32_F (short* target, short* source, short* source_index,
		  int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IXGET32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IXGET32_F(target, source, source_index, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IXPUT_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IXPUT_F (long* target, long* source, long* target_index,
		int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IXPUT_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IXPUT_F(target, source, target_index, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_IXPUT32_F wrapper function 
*** (from getput/gps_getput.c)
******************************************************/
void 
GPSHMEM_IXPUT32_F (short* target, short* source, short* target_index,
		  int* nlong, int* pe)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_IXPUT32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_IXPUT32_F(target, source, target_index, nlong, pe);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FINALIZE_C wrapper function 
*** (from init/gp_init.c)
******************************************************/
void 
GPSHMEM_FINALIZE_C (void)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FINALIZE_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FINALIZE_C();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FINALIZE_CORE_F wrapper function 
*** (from init/gp_init.c)
******************************************************/
void 
GPSHMEM_FINALIZE_CORE_F (void)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FINALIZE_CORE_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FINALIZE_CORE_F();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMALLOC_C wrapper function 
*** (from mem/gp_ma.c)
******************************************************/
void* GPSHMALLOC_C (size_t client_nbytes)
{
  void* retval;
  TAU_PROFILE_TIMER(t, "GPSHMALLOC_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = (void*)PGPSHMALLOC_C(client_nbytes);
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHFREE_C wrapper function 
*** (from mem/gp_ma.c)
******************************************************/
void GPSHFREE_C (void* client_ptr)
{
  TAU_PROFILE_TIMER(t, "GPSHFREE_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHFREE_C(client_ptr);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPPTRALIGN_C wrapper function 
*** (from mem/gp_ma.c)
******************************************************/
int GPPTRALIGN_C (int alignment)
{
  int retval;
  TAU_PROFILE_TIMER(t, "GPPTRALIGN_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = PGPPTRALIGN_C(alignment);
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHPALLOC32_F wrapper function 
*** (from mem/gp_ma.c)
******************************************************/
void 
GPSHPALLOC32_F (void **client_ptr, int *length, int *errcode, int *abort)
{
  TAU_PROFILE_TIMER(t, "GPSHPALLOC32_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHPALLOC32_F(client_ptr, length, errcode, abort);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHPALLOC64_F wrapper function 
*** (from mem/gp_ma.c)
******************************************************/
void 
GPSHPALLOC64_F (void **client_ptr, int *length, int *errcode, int *abort)
{
  TAU_PROFILE_TIMER(t, "GPSHPALLOC64_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHPALLOC64_F(client_ptr, length, errcode, abort);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHFREE_F wrapper function 
*** (from mem/gp_ma.c)
******************************************************/
void 
GPSHFREE_F (void **client_ptr)
{
  TAU_PROFILE_TIMER(t, "GPSHFREE_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHFREE_F(client_ptr);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHPALLOC_F wrapper function 
*** (from mem/gp_ma.c)
******************************************************/
void 
GPSHPALLOC_F (void **client_ptr, int *length, int *errcode, int *abort)
{
  TAU_PROFILE_TIMER(t, "GPSHPALLOC_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHPALLOC_F(client_ptr, length, errcode, abort);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPPTRALIGN_F wrapper function 
*** (from mem/gp_ma.c)
******************************************************/
int 
GPPTRALIGN_F (int* alignment)
{
  int retval;
  TAU_PROFILE_TIMER(t, "GPPTRALIGN_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = PGPPTRALIGN_F(alignment);
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHMALLOC_F wrapper function 
*** (from mem/gp_ma77.c)
******************************************************/
memhandle_t GPSHMALLOC_F (int* type_id, int* length)
{
  memhandle_t retval;
  TAU_PROFILE_TIMER(t, "GPSHMALLOC_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = PGPSHMALLOC_F(type_id, length);
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHINDEX_F wrapper function 
*** (from mem/gp_ma77.c)
******************************************************/
int GPSHINDEX_F (int* type_id, memhandle_t* handle)
{
  int retval;
  TAU_PROFILE_TIMER(t, "GPSHINDEX_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = PGPSHINDEX_F(type_id, handle);
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHMALLOCI_F wrapper function 
*** (from mem/gp_ma77.c)
******************************************************/
memhandle_t GPSHMALLOCI_F (int* type_id, int* length, int* index)
{
  memhandle_t retval;
  TAU_PROFILE_TIMER(t, "GPSHMALLOCI_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = PGPSHMALLOCI_F(type_id, length, index);
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHFREE_HANDLE_F wrapper function 
*** (from mem/gp_ma77.c)
******************************************************/
void GPSHFREE_HANDLE_F (memhandle_t* handle)
{
  TAU_PROFILE_TIMER(t, "GPSHFREE_HANDLE_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHFREE_HANDLE_F(handle);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FENCE_C wrapper function 
*** (from misc/gps_fence.c)
******************************************************/
void 
GPSHMEM_FENCE_C (void)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FENCE_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FENCE_C();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_QUIET_C wrapper function 
*** (from misc/gps_fence.c)
******************************************************/
void 
GPSHMEM_QUIET_C (void)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_QUIET_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_QUIET_C();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FENCE_F wrapper function 
*** (from misc/gps_fence.c)
******************************************************/
void 
GPSHMEM_FENCE_F (void)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FENCE_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FENCE_F();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_QUIET_F wrapper function 
*** (from misc/gps_fence.c)
******************************************************/
void 
GPSHMEM_QUIET_F (void)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_QUIET_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_QUIET_F();
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_SWAP_C wrapper function 
*** (from misc/gps_swap.c)
******************************************************/
int 
GPSHMEM_INT_SWAP_C (int* target, int value, int pe)
{
  int retval;
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_SWAP_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = PGPSHMEM_INT_SWAP_C(target, value, pe);
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHMEM_INT_SWAP_F wrapper function 
*** (from misc/gps_swap.c)
******************************************************/
int 
GPSHMEM_INT_SWAP_F (long* target, long* value, int* pe)
{
  int retval;
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_SWAP_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retval = PGPSHMEM_INT_SWAP_F(target, value, pe);
  TAU_PROFILE_STOP(t);
  return retval;
}


/******************************************************
*** GPSHMEM_WAIT_C wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_WAIT_C (long* ivar, long value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_WAIT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_WAIT_C(ivar, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_WAIT_UNTIL_C wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_WAIT_UNTIL_C (long* ivar, int cmp, long value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_WAIT_UNTIL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_WAIT_UNTIL_C(ivar, cmp, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_WAIT_C wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_INT_WAIT_C (int* ivar, int value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_WAIT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_WAIT_C(ivar, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_WAIT_UNTIL_C wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_INT_WAIT_UNTIL_C (int* ivar, int cmp, int value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_WAIT_UNTIL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_WAIT_UNTIL_C(ivar, cmp, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_WAIT_C wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_LONG_WAIT_C (long* ivar, long value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_WAIT_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_WAIT_C(ivar, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_WAIT_UNTIL_C wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_LONG_WAIT_UNTIL_C (long* ivar, int cmp, long value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_WAIT_UNTIL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_WAIT_UNTIL_C(ivar, cmp, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_WAIT_F wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_WAIT_F (long* ivar, long* value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_WAIT_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_WAIT_F(ivar, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_WAIT_UNTIL_F wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_WAIT_UNTIL_F (long* ivar, int* cmp, long* value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_WAIT_UNTIL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_WAIT_UNTIL_F(ivar, cmp, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_WAIT_F wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_INT_WAIT_F (int* ivar, int* value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_WAIT_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_WAIT_F(ivar, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_WAIT_UNTIL_F wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_INT_WAIT_UNTIL_F (int* ivar, int* cmp, int* value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_WAIT_UNTIL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_WAIT_UNTIL_F(ivar, cmp, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_WAIT_F wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_LONG_WAIT_F (long* ivar, long* value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_WAIT_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_WAIT_F(ivar, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_WAIT_UNTIL_F wrapper function 
*** (from misc/gps_wait.c)
******************************************************/
void 
GPSHMEM_LONG_WAIT_UNTIL_F (long* ivar, int* cmp, long* value)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_WAIT_UNTIL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_WAIT_UNTIL_F(ivar, cmp, value);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_AND_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_AND_TO_ALL__ST_C (int* target, int* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_AND_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_AND_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_AND_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_AND_TO_ALL__ST_C (short* target, short* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_AND_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_AND_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_AND_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_AND_TO_ALL__ST_C (long* target, long* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_AND_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_AND_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_AND_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_AND_TO_ALL__ST_C (long long* target, long long* source,
			int nreduce, int PE_start, int PE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_AND_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_AND_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_AND_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_AND_TO_ALL_C (int* target, int* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_AND_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_AND_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_AND_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_AND_TO_ALL_C (short* target, short* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_AND_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_AND_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_AND_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_AND_TO_ALL_C (long* target, long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_AND_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_AND_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_AND_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_AND_TO_ALL_C (long long* target, long long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_AND_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_AND_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_OR_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_OR_TO_ALL__ST_C (int* target, int* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_OR_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_OR_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_OR_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_OR_TO_ALL__ST_C (short* target, short* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_OR_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_OR_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_OR_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_OR_TO_ALL__ST_C (long* target, long* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_OR_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_OR_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_OR_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_OR_TO_ALL__ST_C (long long* target, long long* source,
			int nreduce, int PE_start, int PE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_OR_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_OR_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_OR_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_OR_TO_ALL_C (int* target, int* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_OR_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_OR_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_OR_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_OR_TO_ALL_C (short* target, short* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_OR_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_OR_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_OR_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_OR_TO_ALL_C (long* target, long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_OR_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_OR_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_OR_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_OR_TO_ALL_C (long long* target, long long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_OR_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_OR_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_XOR_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_XOR_TO_ALL__ST_C (int* target, int* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_XOR_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_XOR_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_XOR_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_XOR_TO_ALL__ST_C (short* target, short* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_XOR_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_XOR_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_XOR_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_XOR_TO_ALL__ST_C (long* target, long* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_XOR_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_XOR_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_XOR_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_XOR_TO_ALL__ST_C (long long* target, long long* source,
			int nreduce, int PE_start, int PE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_XOR_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_XOR_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_XOR_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_XOR_TO_ALL_C (int* target, int* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_XOR_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_XOR_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_XOR_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_XOR_TO_ALL_C (short* target, short* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_XOR_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_XOR_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_XOR_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_XOR_TO_ALL_C (long* target, long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_XOR_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_XOR_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_XOR_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_XOR_TO_ALL_C (long long* target, long long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_XOR_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_XOR_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_SUM_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_SUM_TO_ALL__ST_C (int* target, int* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_SUM_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_SUM_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_SUM_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_SUM_TO_ALL__ST_C (short* target, short* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_SUM_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_SUM_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_SUM_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_SUM_TO_ALL__ST_C (long* target, long* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_SUM_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_SUM_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_SUM_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_SUM_TO_ALL__ST_C (long long* target, long long* source,
			int nreduce, int PE_start, int PE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_SUM_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_SUM_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_SUM_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_SUM_TO_ALL__ST_C (double* target, double* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_SUM_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_SUM_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_SUM_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_SUM_TO_ALL__ST_C (float* target, float* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_SUM_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_SUM_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_SUM_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_SUM_TO_ALL_C (int* target, int* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_SUM_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_SUM_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_SUM_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_SUM_TO_ALL_C (short* target, short* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_SUM_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_SUM_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_SUM_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_SUM_TO_ALL_C (long* target, long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_SUM_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_SUM_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_SUM_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_SUM_TO_ALL_C (long long* target, long long* source,
			int nreduce, int PE_start, int logPE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_SUM_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_SUM_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_SUM_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_SUM_TO_ALL_C (double* target, double* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_SUM_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_SUM_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_SUM_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_SUM_TO_ALL_C (float* target, float* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_SUM_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_SUM_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_PROD_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_PROD_TO_ALL__ST_C (int* target, int* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_PROD_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_PROD_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_PROD_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_PROD_TO_ALL__ST_C (short* target, short* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_PROD_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_PROD_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_PROD_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_PROD_TO_ALL__ST_C (long* target, long* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_PROD_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_PROD_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_PROD_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_PROD_TO_ALL__ST_C (long long* target, long long* source,
			int nreduce, int PE_start, int PE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_PROD_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_PROD_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_PROD_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_PROD_TO_ALL__ST_C (double* target, double* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_PROD_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_PROD_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_PROD_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_PROD_TO_ALL__ST_C (float* target, float* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_PROD_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_PROD_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_PROD_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_PROD_TO_ALL_C (int* target, int* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_PROD_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_PROD_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_PROD_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_PROD_TO_ALL_C (short* target, short* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_PROD_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_PROD_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_PROD_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_PROD_TO_ALL_C (long* target, long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_PROD_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_PROD_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_PROD_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_PROD_TO_ALL_C (long long* target, long long* source,
			int nreduce, int PE_start, int logPE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_PROD_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_PROD_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_PROD_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_PROD_TO_ALL_C (double* target, double* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_PROD_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_PROD_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_PROD_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_PROD_TO_ALL_C (float* target, float* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_PROD_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_PROD_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_MIN_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_MIN_TO_ALL__ST_C (int* target, int* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_MIN_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_MIN_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_MIN_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_MIN_TO_ALL__ST_C (short* target, short* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_MIN_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_MIN_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_MIN_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_MIN_TO_ALL__ST_C (long* target, long* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_MIN_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_MIN_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_MIN_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_MIN_TO_ALL__ST_C (long long* target, long long* source,
			int nreduce, int PE_start, int PE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_MIN_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_MIN_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_MIN_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_MIN_TO_ALL__ST_C (double* target, double* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_MIN_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_MIN_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_MIN_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_MIN_TO_ALL__ST_C (float* target, float* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_MIN_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_MIN_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_MIN_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_MIN_TO_ALL_C (int* target, int* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_MIN_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_MIN_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_MIN_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_MIN_TO_ALL_C (short* target, short* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_MIN_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_MIN_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_MIN_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_MIN_TO_ALL_C (long* target, long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_MIN_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_MIN_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_MIN_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_MIN_TO_ALL_C (long long* target, long long* source,
			int nreduce, int PE_start, int logPE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_MIN_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_MIN_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_MIN_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_MIN_TO_ALL_C (double* target, double* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_MIN_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_MIN_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_MIN_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_MIN_TO_ALL_C (float* target, float* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_MIN_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_MIN_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_MAX_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_MAX_TO_ALL__ST_C (int* target, int* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_MAX_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_MAX_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_MAX_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_MAX_TO_ALL__ST_C (short* target, short* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_MAX_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_MAX_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_MAX_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_MAX_TO_ALL__ST_C (long* target, long* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_MAX_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_MAX_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_MAX_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_MAX_TO_ALL__ST_C (long long* target, long long* source,
			int nreduce, int PE_start, int PE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_MAX_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_MAX_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_MAX_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_MAX_TO_ALL__ST_C (double* target, double* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_MAX_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_MAX_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_MAX_TO_ALL__ST_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_MAX_TO_ALL__ST_C (float* target, float* source, int nreduce,
			int PE_start, int PE_stride, int PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_MAX_TO_ALL__ST_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_MAX_TO_ALL__ST_C(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_MAX_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_MAX_TO_ALL_C (int* target, int* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_MAX_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_MAX_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_MAX_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_MAX_TO_ALL_C (short* target, short* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_MAX_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_MAX_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_MAX_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_MAX_TO_ALL_C (long* target, long* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_MAX_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_MAX_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_MAX_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_MAX_TO_ALL_C (long long* target, long long* source,
			int nreduce, int PE_start, int logPE_stride,
			int PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_MAX_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_MAX_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_MAX_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_MAX_TO_ALL_C (double* target, double* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_MAX_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_MAX_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_MAX_TO_ALL_C wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_MAX_TO_ALL_C (float* target, float* source, int nreduce,
			int PE_start, int logPE_stride, int PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_MAX_TO_ALL_C()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_MAX_TO_ALL_C(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_AND_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_AND_TO_ALL__ST_F (int* target, int* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_AND_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_AND_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_AND_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_AND_TO_ALL__ST_F (short* target, short* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_AND_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_AND_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_AND_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_AND_TO_ALL__ST_F (long* target, long* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_AND_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_AND_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_AND_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_AND_TO_ALL__ST_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* PE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_AND_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_AND_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_AND_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_AND_TO_ALL_F (int* target, int* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_AND_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_AND_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_AND_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_AND_TO_ALL_F (short* target, short* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_AND_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_AND_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_AND_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_AND_TO_ALL_F (long* target, long* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_AND_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_AND_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_AND_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_AND_TO_ALL_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* logPE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_AND_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_AND_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_OR_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_OR_TO_ALL__ST_F (int* target, int* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_OR_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_OR_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_OR_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_OR_TO_ALL__ST_F (short* target, short* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_OR_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_OR_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_OR_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_OR_TO_ALL__ST_F (long* target, long* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_OR_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_OR_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_OR_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_OR_TO_ALL__ST_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* PE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_OR_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_OR_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_OR_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_OR_TO_ALL_F (int* target, int* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_OR_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_OR_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_OR_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_OR_TO_ALL_F (short* target, short* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_OR_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_OR_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_OR_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_OR_TO_ALL_F (long* target, long* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_OR_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_OR_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_OR_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_OR_TO_ALL_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* logPE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_OR_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_OR_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_XOR_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_XOR_TO_ALL__ST_F (int* target, int* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_XOR_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_XOR_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_XOR_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_XOR_TO_ALL__ST_F (short* target, short* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_XOR_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_XOR_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_XOR_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_XOR_TO_ALL__ST_F (long* target, long* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_XOR_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_XOR_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_XOR_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_XOR_TO_ALL__ST_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* PE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_XOR_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_XOR_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_XOR_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_XOR_TO_ALL_F (int* target, int* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_XOR_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_XOR_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_XOR_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_XOR_TO_ALL_F (short* target, short* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_XOR_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_XOR_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_XOR_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_XOR_TO_ALL_F (long* target, long* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_XOR_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_XOR_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_XOR_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_XOR_TO_ALL_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* logPE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_XOR_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_XOR_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_SUM_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_SUM_TO_ALL__ST_F (int* target, int* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_SUM_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_SUM_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_SUM_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_SUM_TO_ALL__ST_F (short* target, short* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_SUM_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_SUM_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_SUM_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_SUM_TO_ALL__ST_F (long* target, long* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_SUM_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_SUM_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_SUM_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_SUM_TO_ALL__ST_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* PE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_SUM_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_SUM_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_SUM_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_SUM_TO_ALL__ST_F (double* target, double* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_SUM_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_SUM_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_SUM_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_SUM_TO_ALL__ST_F (float* target, float* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_SUM_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_SUM_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_SUM_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_SUM_TO_ALL_F (int* target, int* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_SUM_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_SUM_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_SUM_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_SUM_TO_ALL_F (short* target, short* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_SUM_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_SUM_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_SUM_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_SUM_TO_ALL_F (long* target, long* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_SUM_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_SUM_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_SUM_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_SUM_TO_ALL_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* logPE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_SUM_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_SUM_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_SUM_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_SUM_TO_ALL_F (double* target, double* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_SUM_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_SUM_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_SUM_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_SUM_TO_ALL_F (float* target, float* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_SUM_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_SUM_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_PROD_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_PROD_TO_ALL__ST_F (int* target, int* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_PROD_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_PROD_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_PROD_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_PROD_TO_ALL__ST_F (short* target, short* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_PROD_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_PROD_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_PROD_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_PROD_TO_ALL__ST_F (long* target, long* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_PROD_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_PROD_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_PROD_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_PROD_TO_ALL__ST_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* PE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_PROD_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_PROD_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_PROD_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_PROD_TO_ALL__ST_F (double* target, double* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_PROD_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_PROD_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_PROD_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_PROD_TO_ALL__ST_F (float* target, float* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_PROD_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_PROD_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_PROD_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_PROD_TO_ALL_F (int* target, int* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_PROD_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_PROD_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_PROD_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_PROD_TO_ALL_F (short* target, short* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_PROD_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_PROD_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_PROD_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_PROD_TO_ALL_F (long* target, long* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_PROD_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_PROD_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_PROD_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_PROD_TO_ALL_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* logPE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_PROD_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_PROD_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_PROD_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_PROD_TO_ALL_F (double* target, double* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_PROD_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_PROD_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_PROD_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_PROD_TO_ALL_F (float* target, float* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_PROD_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_PROD_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_MIN_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_MIN_TO_ALL__ST_F (int* target, int* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_MIN_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_MIN_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_MIN_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_MIN_TO_ALL__ST_F (short* target, short* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_MIN_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_MIN_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_MIN_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_MIN_TO_ALL__ST_F (long* target, long* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_MIN_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_MIN_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_MIN_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_MIN_TO_ALL__ST_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* PE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_MIN_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_MIN_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_MIN_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_MIN_TO_ALL__ST_F (double* target, double* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_MIN_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_MIN_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_MIN_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_MIN_TO_ALL__ST_F (float* target, float* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_MIN_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_MIN_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_MIN_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_MIN_TO_ALL_F (int* target, int* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_MIN_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_MIN_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_MIN_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_MIN_TO_ALL_F (short* target, short* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_MIN_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_MIN_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_MIN_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_MIN_TO_ALL_F (long* target, long* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_MIN_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_MIN_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_MIN_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_MIN_TO_ALL_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* logPE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_MIN_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_MIN_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_MIN_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_MIN_TO_ALL_F (double* target, double* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_MIN_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_MIN_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_MIN_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_MIN_TO_ALL_F (float* target, float* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_MIN_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_MIN_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_MAX_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_MAX_TO_ALL__ST_F (int* target, int* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_MAX_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_MAX_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_MAX_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_MAX_TO_ALL__ST_F (short* target, short* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_MAX_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_MAX_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_MAX_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_MAX_TO_ALL__ST_F (long* target, long* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_MAX_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_MAX_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_MAX_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_MAX_TO_ALL__ST_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* PE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_MAX_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_MAX_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_MAX_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_MAX_TO_ALL__ST_F (double* target, double* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_MAX_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_MAX_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_MAX_TO_ALL__ST_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_MAX_TO_ALL__ST_F (float* target, float* source, int* nreduce,
			int* PE_start, int* PE_stride, int* PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_MAX_TO_ALL__ST_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_MAX_TO_ALL__ST_F(target, source, nreduce, PE_start, PE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_INT_MAX_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_INT_MAX_TO_ALL_F (int* target, int* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			int* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_INT_MAX_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_INT_MAX_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_SHORT_MAX_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_SHORT_MAX_TO_ALL_F (short* target, short* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			short* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_SHORT_MAX_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_SHORT_MAX_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONG_MAX_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONG_MAX_TO_ALL_F (long* target, long* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONG_MAX_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONG_MAX_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_LONGLONG_MAX_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_LONGLONG_MAX_TO_ALL_F (long long* target, long long* source,
			int* nreduce, int* PE_start, int* logPE_stride,
			int* PE_size, long long* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_LONGLONG_MAX_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_LONGLONG_MAX_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_DOUBLE_MAX_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_DOUBLE_MAX_TO_ALL_F (double* target, double* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			double* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_DOUBLE_MAX_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_DOUBLE_MAX_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


/******************************************************
*** GPSHMEM_FLOAT_MAX_TO_ALL_F wrapper function 
*** (from reduce/gps_reduce.c)
******************************************************/
void 
GPSHMEM_FLOAT_MAX_TO_ALL_F (float* target, float* source, int* nreduce,
			int* PE_start, int* logPE_stride, int* PE_size,
			float* pWrk, long* pSync)
{
  TAU_PROFILE_TIMER(t, "GPSHMEM_FLOAT_MAX_TO_ALL_F()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PGPSHMEM_FLOAT_MAX_TO_ALL_F(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  TAU_PROFILE_STOP(t);
}


