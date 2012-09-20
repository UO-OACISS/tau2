#include <dmapp.h>
#include <Profile/Profiler.h>
#include <stdio.h>

int Tau_my_pe; 
int Tau_total_ranks; 
int tau_dmapp_tagid=0 ;
extern int TAUDECL tau_totalnodes(int set_or_get, int value);
#define dprintf if (0) printf

#define TAU_DMAPP_TAGID (tau_dmapp_tagid = (tau_dmapp_tagid & 255))
#define TAU_DMAPP_TAGID_NEXT ((++tau_dmapp_tagid) & 255)

int Tau_get_dmapp_size(dmapp_type_t data) {
  int size;
  switch (data) {
    case DMAPP_BYTE: /* byte */
      size=1;
      break;
    case DMAPP_DW: /* double word */
      size=4; 
      break;
    case DMAPP_QW: /* quad word */  
      size=8;
      break;
    case DMAPP_DQW: /* double quad word */
      size=16;
      break;
    default: /* default? */
      printf("Tau_get_dmapp_size<%d>: passed unknown default: returning 0 \n", Tau_get_node());
      size=0;
      break;
  }
  return size; 
}
    

/**********************************************************
   dmapp_init
 **********************************************************/

dmapp_return_t  __real_dmapp_init(dmapp_rma_attrs_t * a1, dmapp_rma_attrs_t * a2) ;
dmapp_return_t  __wrap_dmapp_init(dmapp_rma_attrs_t * a1, dmapp_rma_attrs_t * a2) {

  dmapp_return_t retval = 0;

  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_init(dmapp_rma_attrs_t *, dmapp_rma_attrs_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_dmapp_init(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_init_ext
 **********************************************************/

dmapp_return_t  __real_dmapp_init_ext(dmapp_rma_attrs_ext_t * a1, dmapp_rma_attrs_ext_t * a2) ;
dmapp_return_t  __wrap_dmapp_init_ext(dmapp_rma_attrs_ext_t * a1, dmapp_rma_attrs_ext_t * a2) {

  dmapp_return_t retval = 0;
  Tau_create_top_level_timer_if_necessary();

  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_init_ext(dmapp_rma_attrs_ext_t *, dmapp_rma_attrs_ext_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_dmapp_init_ext(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_finalize
 **********************************************************/

dmapp_return_t  __real_dmapp_finalize(void) ;
dmapp_return_t  __wrap_dmapp_finalize() {

  dmapp_return_t retval = 0;

  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_finalize() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_dmapp_finalize();
  TAU_PROFILE_SET_NODE(Tau_my_pe); 
  tau_totalnodes(1,Tau_total_ranks);
  TAU_PROFILE_STOP(t);
  Tau_stop_top_level_timer_if_necessary();
  return retval;

}


/**********************************************************
   dmapp_get_jobinfo
 **********************************************************/

dmapp_return_t  __real_dmapp_get_jobinfo(dmapp_jobinfo_t * a1) ;
dmapp_return_t  __wrap_dmapp_get_jobinfo(dmapp_jobinfo_t * a1) {

  dmapp_return_t retval = 0;
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_get_jobinfo(dmapp_jobinfo_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_dmapp_get_jobinfo(a1);
  TAU_PROFILE_SET_NODE(a1->pe);
  tau_totalnodes(1,a1->npes);
  Tau_my_pe = a1->pe;
  Tau_total_ranks = a1->npes; 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_get_rma_attrs
 **********************************************************/

dmapp_return_t  __real_dmapp_get_rma_attrs(dmapp_rma_attrs_t * a1) ;
dmapp_return_t  __wrap_dmapp_get_rma_attrs(dmapp_rma_attrs_t * a1) {

  dmapp_return_t retval = 0;
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_get_rma_attrs(dmapp_rma_attrs_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_dmapp_get_rma_attrs(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_put_nb
 **********************************************************/

dmapp_return_t  __real_dmapp_put_nb(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, uint64_t a5, dmapp_type_t a6, dmapp_syncid_handle_t * a7) ;
dmapp_return_t  __wrap_dmapp_put_nb(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, uint64_t a5, dmapp_type_t a6, dmapp_syncid_handle_t * a7) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a6) * a5; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_put_nb(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, uint64_t, dmapp_type_t, dmapp_syncid_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  retval  =  __real_dmapp_put_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_put_nbi
 **********************************************************/

dmapp_return_t  __real_dmapp_put_nbi(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, uint64_t a5, dmapp_type_t a6) ;
dmapp_return_t  __wrap_dmapp_put_nbi(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, uint64_t a5, dmapp_type_t a6) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a6) * a5; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_put_nbi(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  TAU_PROFILE_START(t);
  retval  =  __real_dmapp_put_nbi(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_put
 **********************************************************/

dmapp_return_t  __real_dmapp_put(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, uint64_t a5, dmapp_type_t a6) ;
dmapp_return_t  __wrap_dmapp_put(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, uint64_t a5, dmapp_type_t a6) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a6) * a5; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_put(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  TAU_PROFILE_START(t);
  retval  =  __real_dmapp_put(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_get_nb
 **********************************************************/

dmapp_return_t  __real_dmapp_get_nb(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, uint64_t a5, dmapp_type_t a6, dmapp_syncid_handle_t * a7) ;
dmapp_return_t  __wrap_dmapp_get_nb(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, uint64_t a5, dmapp_type_t a6, dmapp_syncid_handle_t * a7) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a6) * a5; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_get_nb(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, uint64_t, dmapp_type_t, dmapp_syncid_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_get_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_get_nbi
 **********************************************************/

dmapp_return_t  __real_dmapp_get_nbi(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, uint64_t a5, dmapp_type_t a6) ;
dmapp_return_t  __wrap_dmapp_get_nbi(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, uint64_t a5, dmapp_type_t a6) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a6) * a5; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_get_nbi(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_get_nbi(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_get
 **********************************************************/

dmapp_return_t  __real_dmapp_get(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, uint64_t a5, dmapp_type_t a6) ;
dmapp_return_t  __wrap_dmapp_get(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, uint64_t a5, dmapp_type_t a6) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a6) * a5; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_get(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_get(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_iput_nb
 **********************************************************/

dmapp_return_t  __real_dmapp_iput_nb(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8, dmapp_syncid_handle_t * a9) ;
dmapp_return_t  __wrap_dmapp_iput_nb(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8, dmapp_syncid_handle_t * a9) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a8) * a7; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_iput_nb(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, ptrdiff_t, ptrdiff_t, uint64_t, dmapp_type_t, dmapp_syncid_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  retval  =  __real_dmapp_iput_nb(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_iput_nbi
 **********************************************************/

dmapp_return_t  __real_dmapp_iput_nbi(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8) ;
dmapp_return_t  __wrap_dmapp_iput_nbi(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a8) * a7; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_iput_nbi(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, ptrdiff_t, ptrdiff_t, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  retval  =  __real_dmapp_iput_nbi(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_iput
 **********************************************************/

dmapp_return_t  __real_dmapp_iput(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8) ;
dmapp_return_t  __wrap_dmapp_iput(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a8) * a7; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_iput(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, ptrdiff_t, ptrdiff_t, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  retval  =  __real_dmapp_iput(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_iget_nb
 **********************************************************/

dmapp_return_t  __real_dmapp_iget_nb(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8, dmapp_syncid_handle_t * a9) ;
dmapp_return_t  __wrap_dmapp_iget_nb(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8, dmapp_syncid_handle_t * a9) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a8) * a7; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_iget_nb(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, ptrdiff_t, ptrdiff_t, uint64_t, dmapp_type_t, dmapp_syncid_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_iget_nb(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_iget_nbi
 **********************************************************/

dmapp_return_t  __real_dmapp_iget_nbi(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8) ;
dmapp_return_t  __wrap_dmapp_iget_nbi(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a8) * a7; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_iget_nbi(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, ptrdiff_t, ptrdiff_t, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_iget_nbi(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_iget
 **********************************************************/

dmapp_return_t  __real_dmapp_iget(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8) ;
dmapp_return_t  __wrap_dmapp_iget(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t a5, ptrdiff_t a6, uint64_t a7, dmapp_type_t a8) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a8) * a7; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_iget(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, ptrdiff_t, ptrdiff_t, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_iget(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_ixput_nb
 **********************************************************/

dmapp_return_t  __real_dmapp_ixput_nb(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7, dmapp_syncid_handle_t * a8) ;
dmapp_return_t  __wrap_dmapp_ixput_nb(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7, dmapp_syncid_handle_t * a8) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_ixput_nb(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, ptrdiff_t *, uint64_t, dmapp_type_t, dmapp_syncid_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  retval  =  __real_dmapp_ixput_nb(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_ixput_nbi
 **********************************************************/

dmapp_return_t  __real_dmapp_ixput_nbi(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7) ;
dmapp_return_t  __wrap_dmapp_ixput_nbi(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_ixput_nbi(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, ptrdiff_t *, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  retval  =  __real_dmapp_ixput_nbi(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_ixput
 **********************************************************/

dmapp_return_t  __real_dmapp_ixput(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7) ;
dmapp_return_t  __wrap_dmapp_ixput(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t a3, void * a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_ixput(void *, dmapp_seg_desc_t *, dmapp_pe_t, void *, ptrdiff_t *, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_DMAPP_TAGID_NEXT, a3, len);
  retval  =  __real_dmapp_ixput(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_DMAPP_TAGID, Tau_get_node(), len, a3); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_ixget_nb
 **********************************************************/

dmapp_return_t  __real_dmapp_ixget_nb(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7, dmapp_syncid_handle_t * a8) ;
dmapp_return_t  __wrap_dmapp_ixget_nb(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7, dmapp_syncid_handle_t * a8) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_ixget_nb(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, ptrdiff_t *, uint64_t, dmapp_type_t, dmapp_syncid_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_ixget_nb(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_ixget_nbi
 **********************************************************/

dmapp_return_t  __real_dmapp_ixget_nbi(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7) ;
dmapp_return_t  __wrap_dmapp_ixget_nbi(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_ixget_nbi(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, ptrdiff_t *, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_ixget_nbi(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_ixget
 **********************************************************/

dmapp_return_t  __real_dmapp_ixget(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7) ;
dmapp_return_t  __wrap_dmapp_ixget(void * a1, void * a2, dmapp_seg_desc_t * a3, dmapp_pe_t a4, ptrdiff_t * a5, uint64_t a6, dmapp_type_t a7) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_ixget(void *, void *, dmapp_seg_desc_t *, dmapp_pe_t, ptrdiff_t *, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_DMAPP_TAGID_NEXT, Tau_get_node(), len, a4); 
  retval  =  __real_dmapp_ixget(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG(TAU_DMAPP_TAGID, a4, len);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_put_ixpe_nb
 **********************************************************/

dmapp_return_t  __real_dmapp_put_ixpe_nb(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t * a3, uint32_t a4, void * a5, uint64_t a6, dmapp_type_t a7, dmapp_syncid_handle_t * a8) ;
dmapp_return_t  __wrap_dmapp_put_ixpe_nb(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t * a3, uint32_t a4, void * a5, uint64_t a6, dmapp_type_t a7, dmapp_syncid_handle_t * a8) {

  dmapp_return_t retval = 0;
  int i; /* iterate through the number of pes in the list */
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_put_ixpe_nb(void *, dmapp_seg_desc_t *, dmapp_pe_t *, uint32_t, void *, uint64_t, dmapp_type_t, dmapp_syncid_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  for (i = 0; i < a4; i ++) { 
  /* a3 contains target_pe_list and a4 is num_target_pes */
    TAU_TRACE_SENDMSG((TAU_DMAPP_TAGID+1+i)&255, a3[i], len);
  }
  retval  =  __real_dmapp_put_ixpe_nb(a1, a2, a3, a4, a5, a6, a7, a8);
  for (i = 0; i < a4; i ++) {
    TAU_TRACE_RECVMSG_REMOTE((TAU_DMAPP_TAGID+1+i)&255, Tau_get_node(), len, a3[i]); 
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_put_ixpe_nbi
 **********************************************************/

dmapp_return_t  __real_dmapp_put_ixpe_nbi(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t * a3, uint32_t a4, void * a5, uint64_t a6, dmapp_type_t a7) ;
dmapp_return_t  __wrap_dmapp_put_ixpe_nbi(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t * a3, uint32_t a4, void * a5, uint64_t a6, dmapp_type_t a7) {

  dmapp_return_t retval = 0;
  int i; /* iterate through the number of pes in the list */
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_put_ixpe_nbi(void *, dmapp_seg_desc_t *, dmapp_pe_t *, uint32_t, void *, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  dprintf("__wrap_dmapp_put_ixpe_nbi:<%d>: a4 = %d, a3[0] = %d\n", Tau_get_node(), a4, a3[0]);
  for (i = 0; i < a4; i ++) { 
  /* a3 contains target_pe_list and a4 is num_target_pes */
    dprintf("__wrap_dmapp_put_ixpe_nbi:<%d>: Before sendmsg: i = %d, a3[i] = %d\n", Tau_get_node(), i, a3[i]);
    TAU_TRACE_SENDMSG((TAU_DMAPP_TAGID+1+i)&255, a3[i], len);
  }
  retval  =  __real_dmapp_put_ixpe_nbi(a1, a2, a3, a4, a5, a6, a7);
  for (i = 0; i < a4; i ++) {
    dprintf("__wrap_dmapp_put_ixpe_nbi:<%d>: i = %d, a3[i] = %d\n", Tau_get_node(), i, a3[i]);
    TAU_TRACE_RECVMSG_REMOTE((TAU_DMAPP_TAGID+1+i)&255, Tau_get_node(), len, a3[i]); 
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   dmapp_put_ixpe
 **********************************************************/

dmapp_return_t  __real_dmapp_put_ixpe(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t * a3, uint32_t a4, void * a5, uint64_t a6, dmapp_type_t a7) ;
dmapp_return_t  __wrap_dmapp_put_ixpe(void * a1, dmapp_seg_desc_t * a2, dmapp_pe_t * a3, uint32_t a4, void * a5, uint64_t a6, dmapp_type_t a7) {

  dmapp_return_t retval = 0;
  uint64_t len = Tau_get_dmapp_size(a7) * a6; 
  int i; /* iterate through the number of pes in the list */
  TAU_PROFILE_TIMER(t,"dmapp_return_t dmapp_put_ixpe(void *, dmapp_seg_desc_t *, dmapp_pe_t *, uint32_t, void *, uint64_t, dmapp_type_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  for (i = 0; i < a4; i ++) { 
  /* a3 contains target_pe_list and a4 is num_target_pes */
    TAU_TRACE_SENDMSG((TAU_DMAPP_TAGID+1+i)&255, a3[i], len);
  }
  retval  =  __real_dmapp_put_ixpe(a1, a2, a3, a4, a5, a6, a7);
  for (i = 0; i < a4; i ++) {
    TAU_TRACE_RECVMSG_REMOTE((TAU_DMAPP_TAGID+1+i)&255, Tau_get_node(), len, a3[i]); 
  }
  TAU_PROFILE_STOP(t);
  return retval;

}

