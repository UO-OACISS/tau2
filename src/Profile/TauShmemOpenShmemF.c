#include <TauPShmemOpenShmemFortran.h>
#include <Profile/Profiler.h>
#include <stdio.h>

int TAUDECL tau_totalnodes(int set_or_get, int value);
static int tau_shmem_tagid_f=0 ; 
#define TAU_SHMEM_TAGID tau_shmem_tagid_f=tau_shmem_tagid_f%250
#define TAU_SHMEM_TAGID_NEXT (++tau_shmem_tagid_f) % 250 

/**********************************************************
   shmem_addr_accessible_
 **********************************************************/

void shmem_addr_accessible_(void * a1, int * a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_addr_accessible_(void *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_addr_accessible_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier_
 **********************************************************/

void shmem_barrier_(int * a1, int * a2, int * a3, long * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_(int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_barrier_(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier_all_
 **********************************************************/

void shmem_barrier_all_()  {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_all_(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_barrier_all_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier_ps_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_barrier_ps_(int * a1, int * a2, int * a3, long * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_ps_(int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_barrier_ps_(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_broadcast32_
 **********************************************************/

void shmem_broadcast32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast32_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast32_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast4_
 **********************************************************/

void shmem_broadcast4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast4_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast4_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast64_
 **********************************************************/

void shmem_broadcast64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast64_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast64_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast8_
 **********************************************************/

void shmem_broadcast8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast8_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast8_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_character_get_
 **********************************************************/

void shmem_character_get_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_character_get_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)* (*a3), (*a4));
   pshmem_character_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(char)* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_character_put_
 **********************************************************/

void shmem_character_put_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_character_put_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(char)* (*a3));
   pshmem_character_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_cache_inv_
 **********************************************************/

void shmem_clear_cache_inv_()  {

  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_inv_(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_clear_cache_inv_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_cache_line_inv_
 **********************************************************/

void shmem_clear_cache_line_inv_(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_line_inv_(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_clear_cache_line_inv_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_lock_
 **********************************************************/

void shmem_clear_lock_(long * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_clear_lock_(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_clear_lock_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_collect4_
 **********************************************************/

void shmem_collect4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_collect4_(void *, void *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_collect4_(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_collect64_
 **********************************************************/

void shmem_collect64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_collect64_(void *, void *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_collect64_(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_collect8_
 **********************************************************/

void shmem_collect8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_collect8_(void *, void *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_collect8_(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_comp4_prod_to_all_
 **********************************************************/

void shmem_comp4_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_comp4_prod_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_comp4_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_comp4_sum_to_all_
 **********************************************************/

void shmem_comp4_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_comp4_sum_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_comp4_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_comp8_prod_to_all_
 **********************************************************/

void shmem_comp8_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_comp8_prod_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_comp8_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_comp8_sum_to_all_
 **********************************************************/

void shmem_comp8_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_comp8_sum_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_comp8_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complex_get_
 **********************************************************/

void shmem_complex_get_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_complex_get_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_complex_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complex_iget_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_complex_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_complex_iget_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
   pshmem_complex_iget_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_complex_iput_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_complex_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_complex_iput_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
   pshmem_complex_iput_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_complex_put_
 **********************************************************/

void shmem_complex_put_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_complex_put_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_complex_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_get_
 **********************************************************/

void shmem_double_get_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_get_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)* (*a3), (*a4));
   pshmem_double_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(double)* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_iget_
 **********************************************************/

void shmem_double_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_iget_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)* (*a5), (*a6));
   pshmem_double_iget_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), sizeof(double)* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_iput_
 **********************************************************/

void shmem_double_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_iput_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), sizeof(double)* (*a5));
   pshmem_double_iput_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put_
 **********************************************************/

void shmem_double_put_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_put_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(double)* (*a3));
   pshmem_double_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fcollect32_
 **********************************************************/

void shmem_fcollect32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_fcollect32_(void *, void *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_fcollect32_(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fcollect4_
 **********************************************************/

void shmem_fcollect4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_fcollect4_(void *, void *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_fcollect4_(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fcollect64_
 **********************************************************/

void shmem_fcollect64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_fcollect64_(void *, void *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_fcollect64_(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fcollect8_
 **********************************************************/

void shmem_fcollect8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_fcollect8_(void *, void *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_fcollect8_(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fence_
 **********************************************************/

void shmem_fence_()  {

  TAU_PROFILE_TIMER(t,"void shmem_fence_(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_fence_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get128_
 **********************************************************/

void shmem_get128_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get128_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16* (*a3), (*a4));
   pshmem_get128_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 16* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get16_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_get16_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get16_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2* (*a3), (*a4));
   pshmem_get16_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 2* (*a3));
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_get32_
 **********************************************************/

void shmem_get32_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get32_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a3), (*a4));
   pshmem_get32_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 4* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get4_
 **********************************************************/

void shmem_get4_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get4_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a3), (*a4));
   pshmem_get4_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 4* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get64_
 **********************************************************/

void shmem_get64_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get64_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a3), (*a4));
   pshmem_get64_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 8* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get8_
 **********************************************************/

void shmem_get8_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get8_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a3), (*a4));
   pshmem_get8_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 8* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_getmem_
 **********************************************************/

void shmem_getmem_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_getmem_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_getmem_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_group_create_strided_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_group_create_strided_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_group_create_strided_(int *, int *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_group_create_strided_(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_group_delete_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_group_delete_(int * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_group_delete_(int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_group_delete_(a1);
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_iget128_
 **********************************************************/

void shmem_iget128_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget128_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16* (*a5), (*a6));
   pshmem_iget128_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 16* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget16_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_iget16_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget16_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2* (*a5), (*a6));
   pshmem_iget16_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 2* (*a5));
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_iget32_
 **********************************************************/

void shmem_iget32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget32_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a5), (*a6));
   pshmem_iget32_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 4* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget4_
 **********************************************************/

void shmem_iget4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget4_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a5), (*a6));
   pshmem_iget4_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 4* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget64_
 **********************************************************/

void shmem_iget64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget64_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a5), (*a6));
   pshmem_iget64_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 8* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget8_
 **********************************************************/

void shmem_iget8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget8_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a5), (*a6));
   pshmem_iget8_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 8* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_and_to_all_
 **********************************************************/

void shmem_int2_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int2_and_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_and_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_max_to_all_
 **********************************************************/

void shmem_int2_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int2_max_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_min_to_all_
 **********************************************************/

void shmem_int2_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int2_min_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_or_to_all_
 **********************************************************/

void shmem_int2_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int2_or_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_or_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_prod_to_all_
 **********************************************************/

void shmem_int2_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int2_prod_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_sum_to_all_
 **********************************************************/

void shmem_int2_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int2_sum_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_xor_to_all_
 **********************************************************/

void shmem_int2_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int2_xor_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_xor_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_add_
 **********************************************************/

void shmem_int4_add_(void * a1, int * a2, int * a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_add_(void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_add_(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_and_to_all_
 **********************************************************/

void shmem_int4_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_and_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_and_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_cswap_
 **********************************************************/

int shmem_int4_cswap_(int * a1, int * a2, int * a3, int * a4)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_cswap_(int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a4));
  retval  =   pshmem_int4_cswap_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)*1);
  if (retval == (*a2)) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a4));
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int4_fadd_
 **********************************************************/

int shmem_int4_fadd_(void * a1, int * a2, int * a3)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_fadd_(void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =   pshmem_int4_fadd_(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int4_finc_
 **********************************************************/

int shmem_int4_finc_(void * a1, int * a2)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_finc_(void *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a2));
  retval  =   pshmem_int4_finc_(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a2), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a2), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a2));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int4_inc_
 **********************************************************/

void shmem_int4_inc_(void * a1, int * a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_inc_(void *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_inc_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_max_to_all_
 **********************************************************/

void shmem_int4_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_max_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_min_to_all_
 **********************************************************/

void shmem_int4_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_min_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_or_to_all_
 **********************************************************/

void shmem_int4_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_or_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_or_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_prod_to_all_
 **********************************************************/

void shmem_int4_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_prod_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_sum_to_all_
 **********************************************************/

void shmem_int4_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_sum_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_swap_
 **********************************************************/

int shmem_int4_swap_(void * a1, int * a2, int * a3)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_swap_(void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =   pshmem_int4_swap_(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int4_wait_
 **********************************************************/

void shmem_int4_wait_(int * a1, int * a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_wait_(int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_wait_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_wait_until_
 **********************************************************/

void shmem_int4_wait_until_(int * a1, int * a2, int * a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_wait_until_(int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_wait_until_(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_xor_to_all_
 **********************************************************/

void shmem_int4_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int4_xor_to_all_(void *, void *, int *, int *, int *, int *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_xor_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_add_
 **********************************************************/

void shmem_int8_add_(void * a1, long * a2, int * a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_add_(void *, long *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_add_(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_and_to_all_
 **********************************************************/

void shmem_int8_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_and_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_and_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_cswap_
 **********************************************************/

long shmem_int8_cswap_(long * a1, long * a2, long * a3, int * a4)  {

  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_cswap_(long *, long *, long *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a4));
  retval  =   pshmem_int8_cswap_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)*1);
  if (retval == (*a2)) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a4));
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int8_fadd_
 **********************************************************/

long shmem_int8_fadd_(void * a1, int * a2, int * a3)  {

  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_fadd_(void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =   pshmem_int8_fadd_(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int8_finc_
 **********************************************************/

long shmem_int8_finc_(void * a1, int * a2)  {

  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_finc_(void *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a2));
  retval  =   pshmem_int8_finc_(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a2), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a2), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a2));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int8_inc_
 **********************************************************/

void shmem_int8_inc_(void * a1, int * a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_inc_(void *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_inc_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_max_to_all_
 **********************************************************/

void shmem_int8_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_max_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_min_to_all_
 **********************************************************/

void shmem_int8_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_min_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_or_to_all_
 **********************************************************/

void shmem_int8_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_or_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_or_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_prod_to_all_
 **********************************************************/

void shmem_int8_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_prod_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_sum_to_all_
 **********************************************************/

void shmem_int8_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_sum_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_swap_
 **********************************************************/

long shmem_int8_swap_(void * a1, long * a2, int * a3)  {

  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_swap_(void *, long *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =   pshmem_int8_swap_(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int8_wait_
 **********************************************************/

void shmem_int8_wait_(long * a1, long * a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_wait_(long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_wait_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_wait_until_
 **********************************************************/

void shmem_int8_wait_until_(long * a1, int * a2, long * a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_wait_until_(long *, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_wait_until_(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_xor_to_all_
 **********************************************************/

void shmem_int8_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int8_xor_to_all_(void *, void *, int *, int *, int *, int *, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_xor_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_integer_get_
 **********************************************************/

void shmem_integer_get_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_integer_get_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)* (*a3), (*a4));
   pshmem_integer_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_integer_iget_
 **********************************************************/

void shmem_integer_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_integer_iget_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)* (*a5), (*a6));
   pshmem_integer_iget_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), sizeof(int)* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_integer_iput_
 **********************************************************/

void shmem_integer_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_integer_iput_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), sizeof(int)* (*a5));
   pshmem_integer_iput_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_integer_put_
 **********************************************************/

void shmem_integer_put_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_integer_put_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)* (*a3));
   pshmem_integer_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput128_
 **********************************************************/

void shmem_iput128_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput128_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 16* (*a5));
   pshmem_iput128_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput16_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_iput16_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput16_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 2* (*a5));
   pshmem_iput16_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_iput32_
 **********************************************************/

void shmem_iput32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput32_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 4* (*a5));
   pshmem_iput32_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput4_
 **********************************************************/

void shmem_iput4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput4_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 4* (*a5));
   pshmem_iput4_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput64_
 **********************************************************/

void shmem_iput64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput64_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 8* (*a5));
   pshmem_iput64_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput8_
 **********************************************************/

void shmem_iput8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput8_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 8* (*a5));
   pshmem_iput8_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_logical_get_
 **********************************************************/

void shmem_logical_get_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_logical_get_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_logical_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_logical_iget_
 **********************************************************/

void shmem_logical_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_logical_iget_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
   pshmem_logical_iget_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_logical_iput_
 **********************************************************/

void shmem_logical_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_logical_iput_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
   pshmem_logical_iput_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_logical_put_
 **********************************************************/

void shmem_logical_put_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_logical_put_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_logical_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemp_my_pe_
 **********************************************************/

int shmemp_my_pe_()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmemp_my_pe_(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   p_my_pe();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmemp_num_pes
 **********************************************************/

int shmemp_num_pes()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmemp_num_pes(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   p_num_pes();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_pe_accessible_
 **********************************************************/

int shmem_pe_accessible_(int * a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_pe_accessible_(int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_pe_accessible_(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_ptr_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_ptr_(void * a1, int * a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_ptr_(void *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_ptr_(a1, a2);
  TAU_PROFILE_STOP(t);

}
#endif /* TAU_OPENSHMEM_EXTENDED */


/**********************************************************
   shmem_put128_
 **********************************************************/

void shmem_put128_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put128_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 16* (*a3));
   pshmem_put128_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put16_
 **********************************************************/

#ifdef TAU_OPENSHMEM_EXTENDED
void shmem_put16_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put16_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 2* (*a3));
   pshmem_put16_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}

#endif /* TAU_OPENSHMEM_EXTENDED */

/**********************************************************
   shmem_put32_
 **********************************************************/

void shmem_put32_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put32_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
   pshmem_put32_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put4_
 **********************************************************/

void shmem_put4_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put4_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
   pshmem_put4_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put64_
 **********************************************************/

void shmem_put64_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put64_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
   pshmem_put64_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put8_
 **********************************************************/

void shmem_put8_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put8_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
   pshmem_put8_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_putmem_
 **********************************************************/

void shmem_putmem_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_putmem_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_putmem_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_quiet_
 **********************************************************/

void shmem_quiet_()  {

  TAU_PROFILE_TIMER(t,"void shmem_quiet_(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_quiet_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real16_max_to_all_
 **********************************************************/

void shmem_real16_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real16_max_to_all_(void *, void *, int *, int *, int *, int *, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real16_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real16_min_to_all_
 **********************************************************/

void shmem_real16_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real16_min_to_all_(void *, void *, int *, int *, int *, int *, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real16_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real16_prod_to_all_
 **********************************************************/

void shmem_real16_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real16_prod_to_all_(void *, void *, int *, int *, int *, int *, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real16_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real16_sum_to_all_
 **********************************************************/

void shmem_real16_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real16_sum_to_all_(void *, void *, int *, int *, int *, int *, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real16_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_max_to_all_
 **********************************************************/

void shmem_real4_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real4_max_to_all_(void *, void *, int *, int *, int *, int *, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real4_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_min_to_all_
 **********************************************************/

void shmem_real4_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real4_min_to_all_(void *, void *, int *, int *, int *, int *, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real4_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_prod_to_all_
 **********************************************************/

void shmem_real4_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real4_prod_to_all_(void *, void *, int *, int *, int *, int *, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real4_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_sum_to_all_
 **********************************************************/

void shmem_real4_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real4_sum_to_all_(void *, void *, int *, int *, int *, int *, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real4_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_swap_
 **********************************************************/

float shmem_real4_swap_(void * a1, float * a2, int * a3)  {

  float retval = 0;
  TAU_PROFILE_TIMER(t,"float shmem_real4_swap_(void *, float *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*1, (*a3));
  retval  =   pshmem_real4_swap_(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 4*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 4*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*1, (*a3));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_real8_max_to_all_
 **********************************************************/

void shmem_real8_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real8_max_to_all_(void *, void *, int *, int *, int *, int *, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real8_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_min_to_all_
 **********************************************************/

void shmem_real8_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real8_min_to_all_(void *, void *, int *, int *, int *, int *, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real8_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_prod_to_all_
 **********************************************************/

void shmem_real8_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real8_prod_to_all_(void *, void *, int *, int *, int *, int *, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real8_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_sum_to_all_
 **********************************************************/

void shmem_real8_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_real8_sum_to_all_(void *, void *, int *, int *, int *, int *, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real8_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_swap_
 **********************************************************/

double shmem_real8_swap_(void * a1, double * a2, int * a3)  {

  double retval = 0;
  TAU_PROFILE_TIMER(t,"double shmem_real8_swap_(void *, double *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*1, (*a3));
  retval  =   pshmem_real8_swap_(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 8*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 8*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*1, (*a3));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_real_get_
 **********************************************************/

void shmem_real_get_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_real_get_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_real_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real_iget_
 **********************************************************/

void shmem_real_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_real_iget_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
   pshmem_real_iget_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real_iput_
 **********************************************************/

void shmem_real_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_real_iput_(void *, void *, int *, int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
   pshmem_real_iput_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real_put_
 **********************************************************/

void shmem_real_put_(void * a1, void * a2, int * a3, int * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_real_put_(void *, void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_real_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_cache_inv_
 **********************************************************/

void shmem_set_cache_inv_()  {

  TAU_PROFILE_TIMER(t,"void shmem_set_cache_inv_(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_set_cache_inv_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_cache_line_inv_
 **********************************************************/

void shmem_set_cache_line_inv_(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_set_cache_line_inv_(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_set_cache_line_inv_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_lock_
 **********************************************************/

void shmem_set_lock_(long * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_set_lock_(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_set_lock_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_swap_
 **********************************************************/

int shmem_swap_(void * a1, int * a2, int * a3)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_swap_(void *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 1, (*a3));
  retval  =   pshmem_swap_(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 1, (*a3));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_test_lock_
 **********************************************************/

int shmem_test_lock_(long * a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_test_lock_(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_test_lock_(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_udcflush_
 **********************************************************/

void shmem_udcflush_()  {

  TAU_PROFILE_TIMER(t,"void shmem_udcflush_(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_udcflush_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_udcflush_line_
 **********************************************************/

void shmem_udcflush_line_(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_udcflush_line_(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_udcflush_line_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_wait_
 **********************************************************/

void shmem_wait_(long * a1, long * a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_wait_(long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_wait_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_wait_until_
 **********************************************************/

void shmem_wait_until_(int * a1, int * a2, int * a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_wait_until_(int *, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_wait_until_(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   start_pes_
 **********************************************************/

void start_pes_(int * a1)  {

  TAU_PROFILE_TIMER(t,"void start_pes_(int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pstart_pes_(a1);
  tau_totalnodes(1,p_num_pes());
  TAU_PROFILE_SET_NODE(p_my_pe());
  TAU_PROFILE_STOP(t);

}

