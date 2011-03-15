
#include <stdio.h>
#include "parmci.h"
#include <TAU.h>


int
ARMCI_AccV (int op, void *scale, armci_giov_t * darr, int len, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_AccV()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 0;
    for (i = 0; i < len; i++)
      bytes += darr[i].ptr_array_len * darr[i].bytes;
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval = PARMCI_AccV (op, scale, darr, len, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


void
ARMCI_Barrier ()
{
  TAU_PROFILE_TIMER(t, "ARMCI_Barrier()", "", TAU_USER);
  TAU_PROFILE_START(t);
  PARMCI_Barrier ();
  TAU_PROFILE_STOP(t);
}


int
ARMCI_AccS (int optype, void *scale, void *src_ptr, int *src_stride_arr,
	    void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels,
	    int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_AccS()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 1;
    for (i = 0; i < stride_levels + 1; i++)
      bytes *= count[i];
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval =
    PARMCI_AccS (optype, scale, src_ptr, src_stride_arr, dst_ptr,
		 dst_stride_arr, count, stride_levels, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


void
ARMCI_Finalize ()
{
  TAU_PROFILE_TIMER(t, "ARMCI_Finalize()", "", TAU_USER);
  TAU_PROFILE_START(t);
  PARMCI_Finalize ();
  TAU_PROFILE_STOP(t);
}


int
ARMCI_NbPut (void *src, void *dst, int bytes, int proc,
	     armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_NbPut()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, bytes);
  rval = PARMCI_NbPut (src, dst, bytes, proc, nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_GetValueInt (void *src, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_GetValueInt()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 4);
  rval = PARMCI_GetValueInt (src, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Put_flag (void *src, void *dst, int bytes, int *f, int v, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Put_flag()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, bytes);
  rval = PARMCI_Put_flag (src, dst, bytes, f, v, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_NbGetS (void *src_ptr, int *src_stride_arr, void *dst_ptr,
	      int *dst_stride_arr, int *count, int stride_levels, int proc,
	      armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_NbGetS()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 1;
    for (i = 0; i < stride_levels + 1; i++)
      bytes *= count[i];
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval =
    PARMCI_NbGetS (src_ptr, src_stride_arr, dst_ptr, dst_stride_arr, count,
		   stride_levels, proc, nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


void *
ARMCI_Malloc_local (armci_size_t bytes)
{
  void *rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Malloc_local()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Malloc_local (bytes);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Free_local (void *ptr)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Free_local()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Free_local (ptr);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Get (void *src, void *dst, int bytes, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Get()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, bytes);
  rval = PARMCI_Get (src, dst, bytes, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Put (void *src, void *dst, int bytes, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Put()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, bytes);
  rval = PARMCI_Put (src, dst, bytes, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Destroy_mutexes ()
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Destroy_mutexes()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Destroy_mutexes ();
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_GetS (void *src_ptr, int *src_stride_arr, void *dst_ptr,
	    int *dst_stride_arr, int *count, int stride_levels, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_GetS()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 1;
    for (i = 0; i < stride_levels + 1; i++)
      bytes *= count[i];
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval =
    PARMCI_GetS (src_ptr, src_stride_arr, dst_ptr, dst_stride_arr, count,
		 stride_levels, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_NbAccV (int op, void *scale, armci_giov_t * darr, int len, int proc,
	      armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_NbAccV()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 0;
    for (i = 0; i < len; i++)
      bytes += darr[i].ptr_array_len * darr[i].bytes;
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval = PARMCI_NbAccV (op, scale, darr, len, proc, nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


float
ARMCI_GetValueFloat (void *src, int proc)
{
  float rval;
  TAU_PROFILE_TIMER(t, "ARMCI_GetValueFloat()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 4);
  rval = PARMCI_GetValueFloat (src, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Malloc (void **ptr_arr, armci_size_t bytes)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Malloc()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Malloc (ptr_arr, bytes);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_NbAccS (int optype, void *scale, void *src_ptr, int *src_stride_arr,
	      void *dst_ptr, int *dst_stride_arr, int *count,
	      int stride_levels, int proc, armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_NbAccS()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 1;
    for (i = 0; i < stride_levels + 1; i++)
      bytes *= count[i];
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval =
    PARMCI_NbAccS (optype, scale, src_ptr, src_stride_arr, dst_ptr,
		   dst_stride_arr, count, stride_levels, proc, nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_PutS (void *src_ptr, int *src_stride_arr, void *dst_ptr,
	    int *dst_stride_arr, int *count, int stride_levels, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_PutS()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 1;
    for (i = 0; i < stride_levels + 1; i++)
      bytes *= count[i];
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval =
    PARMCI_PutS (src_ptr, src_stride_arr, dst_ptr, dst_stride_arr, count,
		 stride_levels, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


void *
ARMCI_Memat (armci_meminfo_t * meminfo, int memflg)
{
  void *rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Memat()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Memat (meminfo, memflg);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_PutV (armci_giov_t * darr, int len, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_PutV()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 0;
    for (i = 0; i < len; i++)
      bytes += darr[i].ptr_array_len * darr[i].bytes;
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval = PARMCI_PutV (darr, len, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Free (void *ptr)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Free()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Free (ptr);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Init_args (int *argc, char ***argv)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Init_args()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Init_args (argc, argv);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_PutValueInt (int src, void *dst, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_PutValueInt()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 4);
  rval = PARMCI_PutValueInt (src, dst, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


void
ARMCI_Memget (size_t bytes, armci_meminfo_t * meminfo, int memflg)
{
  TAU_PROFILE_TIMER(t, "ARMCI_Memget()", "", TAU_USER);
  TAU_PROFILE_START(t);
  PARMCI_Memget (bytes, meminfo, memflg);
  TAU_PROFILE_STOP(t);
}


void
ARMCI_AllFence ()
{
  TAU_PROFILE_TIMER(t, "ARMCI_AllFence()", "", TAU_USER);
  TAU_PROFILE_START(t);
  PARMCI_AllFence ();
  TAU_PROFILE_STOP(t);
}


int
ARMCI_NbPutV (armci_giov_t * darr, int len, int proc, armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_NbPutV()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 0;
    for (i = 0; i < len; i++)
      bytes += darr[i].ptr_array_len * darr[i].bytes;
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval = PARMCI_NbPutV (darr, len, proc, nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_PutValueDouble (double src, void *dst, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_PutValueDouble()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 8);
  rval = PARMCI_PutValueDouble (src, dst, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_GetV (armci_giov_t * darr, int len, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_GetV()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 0;
    for (i = 0; i < len; i++)
      bytes += darr[i].ptr_array_len * darr[i].bytes;
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval = PARMCI_GetV (darr, len, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Test (armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Test()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Test (nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


void
ARMCI_Unlock (int mutex, int proc)
{
  TAU_PROFILE_TIMER(t, "ARMCI_Unlock()", "", TAU_USER);
  TAU_PROFILE_START(t);
  PARMCI_Unlock (mutex, proc);
  TAU_PROFILE_STOP(t);
}


void
ARMCI_Fence (int proc)
{
  TAU_PROFILE_TIMER(t, "ARMCI_Fence()", "", TAU_USER);
  TAU_PROFILE_START(t);
  PARMCI_Fence (proc);
  TAU_PROFILE_STOP(t);
}


int
ARMCI_Create_mutexes (int num)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Create_mutexes()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Create_mutexes (num);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_PutS_flag (void *src_ptr, int *src_stride_arr, void *dst_ptr,
		 int *dst_stride_arr, int *count, int stride_levels,
		 int *flag, int val, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_PutS_flag()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 1;
    for (i = 0; i < stride_levels + 1; i++)
      bytes *= count[i];
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval =
    PARMCI_PutS_flag (src_ptr, src_stride_arr, dst_ptr, dst_stride_arr, count,
		      stride_levels, flag, val, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_WaitProc (int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_WaitProc()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_WaitProc (proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


void
ARMCI_Lock (int mutex, int proc)
{
  TAU_PROFILE_TIMER(t, "ARMCI_Lock()", "", TAU_USER);
  TAU_PROFILE_START(t);
  PARMCI_Lock (mutex, proc);
  TAU_PROFILE_STOP(t);
}


double
ARMCI_GetValueDouble (void *src, int proc)
{
  double rval;
  TAU_PROFILE_TIMER(t, "ARMCI_GetValueDouble()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 8);
  rval = PARMCI_GetValueDouble (src, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_NbGetV (armci_giov_t * darr, int len, int proc, armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_NbGetV()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 0;
    for (i = 0; i < len; i++)
      bytes += darr[i].ptr_array_len * darr[i].bytes;
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval = PARMCI_NbGetV (darr, len, proc, nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Rmw (int op, void *ploc, void *prem, int extra, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Rmw()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 4);
  rval = PARMCI_Rmw (op, ploc, prem, extra, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Init ()
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Init()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Init ();
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_WaitAll ()
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_WaitAll()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_WaitAll ();
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_NbGet (void *src, void *dst, int bytes, int proc,
	     armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_NbGet()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, bytes);
  rval = PARMCI_NbGet (src, dst, bytes, proc, nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_PutValueFloat (float src, void *dst, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_PutValueFloat()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 4);
  rval = PARMCI_PutValueFloat (src, dst, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_NbPutS (void *src_ptr, int *src_stride_arr, void *dst_ptr,
	      int *dst_stride_arr, int *count, int stride_levels, int proc,
	      armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_NbPutS()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 1;
    for (i = 0; i < stride_levels + 1; i++)
      bytes *= count[i];
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval =
    PARMCI_NbPutS (src_ptr, src_stride_arr, dst_ptr, dst_stride_arr, count,
		   stride_levels, proc, nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_PutS_flag_dir (void *src_ptr, int *src_stride_arr, void *dst_ptr,
		     int *dst_stride_arr, int *count, int stride_levels,
		     int *flag, int val, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_PutS_flag_dir()", "", TAU_USER);
  TAU_PROFILE_START(t);
  {
    int i, bytes = 1;
    for (i = 0; i < stride_levels + 1; i++)
      bytes *= count[i];
    TAU_TRACE_SENDMSG (1, proc, bytes);
  }
  rval =
    PARMCI_PutS_flag_dir (src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
			  count, stride_levels, flag, val, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_PutValueLong (long src, void *dst, int proc)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_PutValueLong()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 8);
  rval = PARMCI_PutValueLong (src, dst, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}


int
ARMCI_Wait (armci_hdl_t * nb_handle)
{
  int rval;
  TAU_PROFILE_TIMER(t, "ARMCI_Wait()", "", TAU_USER);
  TAU_PROFILE_START(t);
  rval = PARMCI_Wait (nb_handle);
  TAU_PROFILE_STOP(t);
  return rval;
}


long
ARMCI_GetValueLong (void *src, int proc)
{
  long rval;
  TAU_PROFILE_TIMER(t, "ARMCI_GetValueLong()", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG (1, proc, 8);
  rval = PARMCI_GetValueLong (src, proc);
  TAU_PROFILE_STOP(t);
  return rval;
}
