#include <stdio.h>
#include <Profile/TauPShmemCrayFortran.h>
#include <Profile/Profiler.h>

int TAUDECL tau_totalnodes(int set_or_get, int value);
static int tau_shmem_tagid_f=0 ; 
#define TAU_SHMEM_TAGID tau_shmem_tagid_f=tau_shmem_tagid_f%250
#define TAU_SHMEM_TAGID_NEXT (++tau_shmem_tagid_f) % 250 

/**********************************************************
   shmem_broadcast4_
 **********************************************************/

void shmem_broadcast4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast4_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast4_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get4_
 **********************************************************/

void shmem_get4_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get4_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a3), (*a4));
   pshmem_get4_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 4* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_and_to_all_
 **********************************************************/

void shmem_int4_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_and_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_and_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_cswap_
 **********************************************************/

int shmem_int4_cswap_(void * a1, int * a2, int * a3, int * a4) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int4_cswap_(void *, int *, int *, int *)", "", TAU_USER);
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

int shmem_int4_fadd_(void * a1, int * a2, int * a3) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int4_fadd_(void *, int *, int *)", "", TAU_USER);
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

int shmem_int4_finc_(void * a1, int * a2) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int4_finc_(void *, int *)", "", TAU_USER);
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
   shmem_int4_max_to_all_
 **********************************************************/

void shmem_int4_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_max_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_min_to_all_
 **********************************************************/

void shmem_int4_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_min_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_or_to_all_
 **********************************************************/

void shmem_int4_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_or_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_or_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_prod_to_all_
 **********************************************************/

void shmem_int4_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_prod_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_sum_to_all_
 **********************************************************/

void shmem_int4_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_sum_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_swap_
 **********************************************************/

int shmem_int4_swap_(void * a1, int * a2, int * a3) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int4_swap_(void *, int *, int *)", "", TAU_USER);
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

void shmem_int4_wait_(int * a1, int * a2) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_wait_(int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_wait_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_wait_until_
 **********************************************************/

void shmem_int4_wait_until_(int * a1, int * a2, int * a3) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_wait_until_(int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_wait_until_(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int4_xor_to_all_
 **********************************************************/

void shmem_int4_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int4_xor_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int4_xor_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put4_
 **********************************************************/

void shmem_put4_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put4_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
   pshmem_put4_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put4_nb_
 **********************************************************/

void shmem_put4_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put4_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
   pshmem_put4_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_max_to_all_
 **********************************************************/

void shmem_real4_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_real4_max_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real4_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_min_to_all_
 **********************************************************/

void shmem_real4_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_real4_min_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real4_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_prod_to_all_
 **********************************************************/

void shmem_real4_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_real4_prod_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real4_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_sum_to_all_
 **********************************************************/

void shmem_real4_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_real4_sum_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real4_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real4_swap_
 **********************************************************/

void shmem_real4_swap_(void * a1, void * a2, int * a3) {

  TAU_PROFILE_TIMER(t,"void shmem_real4_swap_(void *, void *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*1, (*a3));
   pshmem_real4_swap_(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 4*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 4*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*1, (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast8_
 **********************************************************/

void shmem_broadcast8_(void * a1, void * a2, long * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast8_(void *, void *, long *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast8_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get8_
 **********************************************************/

void shmem_get8_(void * a1, void * a2, long * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get8_(void *, void *, long *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a3), (*a4));
   pshmem_get8_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 8* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_and_to_all_
 **********************************************************/

void shmem_int8_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_and_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_and_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_cswap_
 **********************************************************/

long shmem_int8_cswap_(void * a1, long * a2, long * a3, int * a4) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_int8_cswap_(void *, long *, long *, int *)", "", TAU_USER);
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

long shmem_int8_fadd_(void * a1, long * a2, int * a3) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_int8_fadd_(void *, long *, int *)", "", TAU_USER);
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

long shmem_int8_finc_(void * a1, int * a2) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_int8_finc_(void *, int *)", "", TAU_USER);
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
   shmem_int8_max_to_all_
 **********************************************************/

void shmem_int8_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_max_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_min_to_all_
 **********************************************************/

void shmem_int8_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_min_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_or_to_all_
 **********************************************************/

void shmem_int8_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_or_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_or_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_prod_to_all_
 **********************************************************/

void shmem_int8_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_prod_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_sum_to_all_
 **********************************************************/

void shmem_int8_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_sum_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_swap_
 **********************************************************/

long shmem_int8_swap_(void * a1, long * a2, int * a3) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_int8_swap_(void *, long *, int *)", "", TAU_USER);
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

void shmem_int8_wait_(long * a1, long * a2) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_wait_(long *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_wait_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_wait_until_
 **********************************************************/

void shmem_int8_wait_until_(long * a1, int * a2, long * a3) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_wait_until_(long *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_wait_until_(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int8_xor_to_all_
 **********************************************************/

void shmem_int8_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int8_xor_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int8_xor_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put8_
 **********************************************************/

void shmem_put8_(void * a1, void * a2, long * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put8_(void *, void *, long *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
   pshmem_put8_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put8_nb_
 **********************************************************/

void shmem_put8_nb_(void * a1, void * a2, long * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put8_nb_(void *, void *, long *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
   pshmem_put8_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_max_to_all_
 **********************************************************/

void shmem_real8_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_real8_max_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real8_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_min_to_all_
 **********************************************************/

void shmem_real8_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_real8_min_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real8_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_prod_to_all_
 **********************************************************/

void shmem_real8_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_real8_prod_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real8_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_sum_to_all_
 **********************************************************/

void shmem_real8_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_real8_sum_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_real8_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real8_swap_
 **********************************************************/

double shmem_real8_swap_(void * a1, void * a2, int * a3) {

  double retval;
  TAU_PROFILE_TIMER(t,"double shmem_real8_swap_(void *, void *, int *)", "", TAU_USER);
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
   shmem_barrier_
 **********************************************************/

void shmem_barrier_(int * a1, int * a2, int * a3, long * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_(int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_barrier_(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier_all_
 **********************************************************/

void shmem_barrier_all_() {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_all_()", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_barrier_all_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast32_
 **********************************************************/

void shmem_broadcast32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast32_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast32_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast64_
 **********************************************************/

void shmem_broadcast64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast64_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast64_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast_
 **********************************************************/

void shmem_broadcast_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_character_get_
 **********************************************************/

void shmem_character_get_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_character_get_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)* (*a3), (*a4));
   pshmem_character_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(char)* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_character_put_
 **********************************************************/

void shmem_character_put_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_character_put_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(char)* (*a3));
   pshmem_character_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_event_
 **********************************************************/

void shmem_clear_event_(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_clear_event_(long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_clear_event_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_lock_
 **********************************************************/

void shmem_clear_lock_(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_clear_lock_(long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_clear_lock_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complex_get_
 **********************************************************/

void shmem_complex_get_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_complex_get_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_complex_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complex_put_
 **********************************************************/

void shmem_complex_put_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_complex_put_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_complex_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complex_put_nb_
 **********************************************************/

void shmem_complex_put_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_complex_put_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_complex_put_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_get_
 **********************************************************/

void shmem_double_get_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_double_get_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)* (*a3), (*a4));
   pshmem_double_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(double)* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put_
 **********************************************************/

void shmem_double_put_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_double_put_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(double)* (*a3));
   pshmem_double_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put_nb_
 **********************************************************/

void shmem_double_put_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_double_put_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(double)* (*a3));
   pshmem_double_put_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fence_
 **********************************************************/

void shmem_fence_() {

  TAU_PROFILE_TIMER(t,"void shmem_fence_()", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_fence_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_finalize_
 **********************************************************/

void shmem_finalize_() {

  TAU_PROFILE_TIMER(t,"void shmem_finalize_()", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_finalize_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get128_
 **********************************************************/

void shmem_get128_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get128_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16* (*a3), (*a4));
   pshmem_get128_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 16* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get16_
 **********************************************************/

void shmem_get16_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get16_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2* (*a3), (*a4));
   pshmem_get16_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 2* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get32_
 **********************************************************/

void shmem_get32_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get32_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a3), (*a4));
   pshmem_get32_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 4* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get64_
 **********************************************************/

void shmem_get64_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get64_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a3), (*a4));
   pshmem_get64_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 8* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get_
 **********************************************************/

void shmem_get_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_getmem_
 **********************************************************/

void shmem_getmem_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_getmem_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_getmem_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget128_
 **********************************************************/

void shmem_iget128_(void * a1, void * a2, void * a3, int * a4, int * a5, int * a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget128_(void *, void *, void *, int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16* (*a5), (*a6));
   pshmem_iget128_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 16* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget16_
 **********************************************************/

void shmem_iget16_(void * a1, void * a2, void * a3, int * a4, int * a5, int * a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget16_(void *, void *, void *, int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2* (*a5), (*a6));
   pshmem_iget16_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 2* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget32_
 **********************************************************/

void shmem_iget32_(void * a1, void * a2, void * a3, int * a4, int * a5, int * a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget32_(void *, void *, void *, int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a5), (*a6));
   pshmem_iget32_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 4* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget64_
 **********************************************************/

void shmem_iget64_(void * a1, void * a2, void * a3, int * a4, int * a5, int * a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget64_(void *, void *, void *, int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a5), (*a6));
   pshmem_iget64_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 8* (*a5));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_init_
 **********************************************************/

void shmem_init_() {

  TAU_PROFILE_TIMER(t,"void shmem_init_()", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_init_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_and_to_all_
 **********************************************************/

void shmem_int2_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int2_and_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_and_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_max_to_all_
 **********************************************************/

void shmem_int2_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int2_max_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_max_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_min_to_all_
 **********************************************************/

void shmem_int2_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int2_min_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_min_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_or_to_all_
 **********************************************************/

void shmem_int2_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int2_or_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_or_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_prod_to_all_
 **********************************************************/

void shmem_int2_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int2_prod_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_prod_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_sum_to_all_
 **********************************************************/

void shmem_int2_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int2_sum_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_sum_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int2_xor_to_all_
 **********************************************************/

void shmem_int2_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, void * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int2_xor_to_all_(void *, void *, int *, int *, int *, int *, void *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int2_xor_to_all_(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_integer_get_
 **********************************************************/

void shmem_integer_get_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_integer_get_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)* (*a3), (*a4));
   pshmem_integer_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)* (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_integer_put_
 **********************************************************/

void shmem_integer_put_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_integer_put_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)* (*a3));
   pshmem_integer_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_integer_put_nb_
 **********************************************************/

void shmem_integer_put_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_integer_put_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)* (*a3));
   pshmem_integer_put_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput128_
 **********************************************************/

void shmem_iput128_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput128_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 16* (*a5));
   pshmem_iput128_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput16_
 **********************************************************/

void shmem_iput16_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput16_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 2* (*a5));
   pshmem_iput16_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput32_
 **********************************************************/

void shmem_iput32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput32_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 4* (*a5));
   pshmem_iput32_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput64_
 **********************************************************/

void shmem_iput64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput64_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 8* (*a5));
   pshmem_iput64_(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a5), (*a6));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_logical_get_
 **********************************************************/

void shmem_logical_get_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_logical_get_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_logical_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_logical_put_
 **********************************************************/

void shmem_logical_put_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_logical_put_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_logical_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_logical_put_nb_
 **********************************************************/

void shmem_logical_put_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_logical_put_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_logical_put_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   start_pes_
 **********************************************************/

void start_pes_(int * a1) {

  TAU_PROFILE_TIMER(t,"void start_pes_(int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pstart_pes_(a1);
  tau_totalnodes(1,pshmem_n_pes());
  TAU_PROFILE_SET_NODE(pshmem_my_pe());
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_my_pe_
 **********************************************************/

int shmem_my_pe_() {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_my_pe_()", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_my_pe_();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_n_pes_
 **********************************************************/

int shmem_n_pes_() {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_n_pes_()", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_n_pes_();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_put128_
 **********************************************************/

void shmem_put128_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put128_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 16* (*a3));
   pshmem_put128_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put128_nb_
 **********************************************************/

void shmem_put128_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put128_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 16* (*a3));
   pshmem_put128_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put16_
 **********************************************************/

void shmem_put16_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put16_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 2* (*a3));
   pshmem_put16_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put16_nb_
 **********************************************************/

void shmem_put16_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put16_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 2* (*a3));
   pshmem_put16_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put32_
 **********************************************************/

void shmem_put32_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put32_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
   pshmem_put32_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put32_nb_
 **********************************************************/

void shmem_put32_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put32_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
   pshmem_put32_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put64_
 **********************************************************/

void shmem_put64_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put64_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
   pshmem_put64_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put64_nb_
 **********************************************************/

void shmem_put64_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put64_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
   pshmem_put64_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put_
 **********************************************************/

void shmem_put_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put_nb_
 **********************************************************/

void shmem_put_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_put_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_putmem_
 **********************************************************/

void shmem_putmem_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_putmem_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_putmem_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_putmem_nb_
 **********************************************************/

void shmem_putmem_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_putmem_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_putmem_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_quiet_
 **********************************************************/

void shmem_quiet_() {

  TAU_PROFILE_TIMER(t,"void shmem_quiet_()", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_quiet_();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real_get_
 **********************************************************/

void shmem_real_get_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_real_get_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
   pshmem_real_get_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real_put_
 **********************************************************/

void shmem_real_put_(void * a1, void * a2, int * a3, int * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_real_put_(void *, void *, int *, int *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_real_put_(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_real_put_nb_
 **********************************************************/

void shmem_real_put_nb_(void * a1, void * a2, int * a3, int * a4, void * a5) {

  TAU_PROFILE_TIMER(t,"void shmem_real_put_nb_(void *, void *, int *, int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
   pshmem_real_put_nb_(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_event_
 **********************************************************/

void shmem_set_event_(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_set_event_(long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_set_event_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_lock_
 **********************************************************/

void shmem_set_lock_(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_set_lock_(long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_set_lock_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_test_event_
 **********************************************************/

int shmem_test_event_(long * a1) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_test_event_(long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_test_event_(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_test_lock_
 **********************************************************/

int shmem_test_lock_(long * a1) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_test_lock_(long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_test_lock_(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_wait_
 **********************************************************/

void shmem_wait_(int * a1, void * a2) {

  TAU_PROFILE_TIMER(t,"void shmem_wait_(int *, void *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_wait_(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_wait_event_
 **********************************************************/

void shmem_wait_event_(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_wait_event_(long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_wait_event_(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_wait_until_
 **********************************************************/

void shmem_wait_until_(void * a1, int * a2, long * a3) {

  TAU_PROFILE_TIMER(t,"void shmem_wait_until_(void *, int *, long *)", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_wait_until_(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}

