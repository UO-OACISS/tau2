/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.acl.lanl.gov/tau                        **
*****************************************************************************
**    Copyright 1997                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : TauFMpi.cpp                                     **
**      Description     : TAU Profiling Package MPI wrapper for F77/F90   **
**      Author          : Sameer Shende                                   **
**      Contact         : sameer@cs.uoregon.edu sameer@acl.lanl.gov       **
**      Flags           : Compile with                                    **
**                        -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**                        -DPROFILE_STATS for Std. Deviation of Excl Time **
**                        -DSGI_HW_COUNTERS for using SGI counters        **
**                        -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
**                        -DTULIP_TIMERS for non-sgi Platform             **
**                        -DPOOMA_STDSTL for using STD STL in POOMA src   **
**                        -DPOOMA_TFLOP for Intel Teraflop at SNL/NM      **
**                        -DPOOMA_KAI for KCC compiler                    **
**                        -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**      Documentation   : See http://www.acl.lanl.gov/tau                 **
***************************************************************************/

#include <mpi.h>

void  mpi_allgather_( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr)
void * sendbuf;
int *sendcount;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcount;
MPI_Datatype *recvtype;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Allgather( sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, *recvtype, *comm );

}

void  mpi_allgatherv_( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr)
void * sendbuf;
int *sendcount;
MPI_Datatype *sendtype;
void * recvbuf;
int * recvcounts;
int * displs;
MPI_Datatype *recvtype;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Allgatherv( sendbuf, *sendcount, *sendtype, recvbuf, recvcounts, displs, *recvtype, *comm );

}

void   mpi_allreduce_( sendbuf, recvbuf, count, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
int *count;
MPI_Datatype *datatype;
MPI_Op *op;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Allreduce( sendbuf, recvbuf, *count, *datatype, *op, *comm );

}

void  mpi_alltoall_( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr)
void * sendbuf;
int *sendcount;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcnt;
MPI_Datatype *recvtype;
MPI_Comm *comm;
int *ierr; 
{
  *ierr = MPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcnt, *recvtype, *comm );
}

void   mpi_alltoallv_( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr)
void * sendbuf;
int * sendcnts;
int * sdispls;
MPI_Datatype *sendtype;
void * recvbuf;
int * recvcnts;
int * rdispls;
MPI_Datatype *recvtype;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Alltoallv( sendbuf, sendcnts, sdispls, *sendtype, recvbuf, recvcnts, rdispls, *recvtype, *comm );

}

void   mpi_barrier_( comm , ierr)
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Barrier( *comm );
}

void   mpi_bcast_( buffer, count, datatype, root, comm , ierr)
void * buffer;
int *count;
MPI_Datatype *datatype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Bcast( buffer, *count, *datatype, *root, *comm );

}

void  mpi_gather_( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr)
void * sendbuf;
int *sendcnt;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcount;
MPI_Datatype *recvtype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  
  *ierr = MPI_Gather( sendbuf, *sendcnt, *sendtype, recvbuf, *recvcount, *recvtype, *root, *comm );

}

void mpi_gatherv_( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr)
void * sendbuf;
int *sendcnt;
MPI_Datatype *sendtype;
void * recvbuf;
int * recvcnts;
int * displs;
MPI_Datatype *recvtype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Gatherv( sendbuf, *sendcnt, *sendtype, recvbuf, recvcnts, displs, *recvtype, *root, *comm );

}

void mpi_op_create_( function, commute, op , ierr)
MPI_User_function * function;
int *commute;
MPI_Op * op;
int *ierr;
{
  *ierr = MPI_Op_create( function, *commute, op );

}

void  mpi_op_free_( op , ierr)
MPI_Op * op;
int *ierr;
{
  *ierr = MPI_Op_free( op );
}

void mpi_reduce_scatter_( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
int * recvcnts;
MPI_Datatype *datatype;
MPI_Op *op;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Reduce_scatter( sendbuf, recvbuf, recvcnts, *datatype, *op, *comm );
}

void mpi_reduce_( sendbuf, recvbuf, count, datatype, op, root, comm , ierr)
void * sendbuf;
void * recvbuf;
int *count;
MPI_Datatype *datatype;
MPI_Op *op;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Reduce( sendbuf, recvbuf, *count, *datatype, *op, *root, *comm );
}

void mpi_scan_( sendbuf, recvbuf, count, datatype, op, comm, ierr)
void * sendbuf;
void * recvbuf;
int *count;
MPI_Datatype *datatype;
MPI_Op *op;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Scan( sendbuf, recvbuf, *count, *datatype, *op, *comm );
}

void   mpi_scatter_( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr )
void * sendbuf;
int *sendcnt;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcnt;
MPI_Datatype *recvtype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Scatter( sendbuf, *sendcnt, *sendtype, recvbuf, *recvcnt, *recvtype, *root, *comm );
}

void  mpi_scatterv_( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr)
void * sendbuf;
int * sendcnts;
int * displs;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcnt;
MPI_Datatype *recvtype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Scatterv( sendbuf, sendcnts, displs, *sendtype, recvbuf, *recvcnt, *recvtype, *root, *comm );
}

void   mpi_attr_delete_( comm, keyval, ierr)
MPI_Comm *comm;
int *keyval;
int *ierr;
{
  *ierr = MPI_Attr_delete( *comm, *keyval );
}

void mpi_attr_get_( comm, keyval, attr_value, flag , ierr)
MPI_Comm *comm;
int *keyval;
void * attr_value;
int * flag;
int *ierr;
{
  *ierr = MPI_Attr_get( *comm, *keyval, attr_value, flag );
}

void   mpi_attr_put_( comm, keyval, attr_value, ierr)
MPI_Comm *comm;
int *keyval;
void * attr_value;
int *ierr;
{
  *ierr = MPI_Attr_put( *comm, *keyval, attr_value );
}

void  mpi_comm_compare_( comm1, comm2, result, ierr )
MPI_Comm *comm1;
MPI_Comm *comm2;
int * result;
int *ierr;
{
  *ierr = MPI_Comm_compare( *comm1, *comm2, result );
}

void  mpi_comm_create_( comm, group, comm_out, ierr )
MPI_Comm *comm;
MPI_Group *group;
MPI_Comm * comm_out;
int *ierr;
{
  *ierr = MPI_Comm_create( *comm, *group, comm_out );
}

void   mpi_comm_dup_( comm, comm_out, ierr )
MPI_Comm *comm;
MPI_Comm * comm_out;
int *ierr;
{
  *ierr = MPI_Comm_dup( *comm, comm_out );
}

void   mpi_comm_free_( comm, ierr)
MPI_Comm * comm;
int *ierr;
{
  *ierr = MPI_Comm_free( comm );
}

void   mpi_comm_group_( comm, group, ierr )
MPI_Comm *comm;
MPI_Group * group;
int *ierr;
{
  *ierr = MPI_Comm_group( *comm, group );
}

void   mpi_comm_rank_( comm, rank, ierr )
MPI_Comm *comm;
int * rank;
int *ierr;
{
  *ierr = MPI_Comm_rank( *comm, rank );
}

void   mpi_comm_remote_group_( comm, group, ierr )
MPI_Comm *comm;
MPI_Group * group;
int *ierr;
{
  *ierr = MPI_Comm_remote_group( *comm, group );
}

void   mpi_comm_remote_size_( comm, size, ierr )
MPI_Comm *comm;
int * size;
int *ierr;
{
  *ierr = MPI_Comm_remote_size( *comm, size );
}

void   mpi_comm_size_( comm, size , ierr)
MPI_Comm *comm;
int * size;
int *ierr;
{
  *ierr = MPI_Comm_size( *comm, size );
}

void   mpi_comm_split_( comm, color, key, comm_out, ierr )
MPI_Comm *comm;
int *color;
int *key;
MPI_Comm * comm_out;
int *ierr;
{
  MPI_Comm l_comm_out;
  *ierr = MPI_Comm_split( *comm, *color, *key, &l_comm_out );
  *comm_out = l_comm_out; 
}

void   mpi_comm_test_inter_( comm, flag, ierr )
MPI_Comm *comm;
int * flag;
int *ierr;
{
  *ierr = MPI_Comm_test_inter( *comm, flag );
}

void   mpi_group_compare_( group1, group2, result, ierr )
MPI_Group *group1;
MPI_Group *group2;
int * result;
int *ierr;
{
  *ierr = MPI_Group_compare( *group1, *group2, result );
}

void   mpi_group_difference_( group1, group2, group_out, ierr )
MPI_Group *group1;
MPI_Group *group2;
MPI_Group * group_out;
int *ierr;
{
  *ierr = MPI_Group_difference( *group1, *group2, group_out );
}

void   mpi_group_excl_( group, n, ranks, newgroup, ierr )
MPI_Group *group;
int *n;
int * ranks;
MPI_Group * newgroup;
int *ierr;
{
  *ierr = MPI_Group_excl( *group, *n, ranks, newgroup );
}

void   mpi_group_free_( group, ierr)
MPI_Group * group;
int *ierr;
{
  *ierr = MPI_Group_free( group );
}

void   mpi_group_incl_( group, n, ranks, group_out, ierr )
MPI_Group *group;
int *n;
int * ranks;
MPI_Group * group_out;
int *ierr;
{
  *ierr = MPI_Group_incl( *group, *n, ranks, group_out );
}

void   mpi_group_intersection_( group1, group2, group_out, ierr )
MPI_Group *group1;
MPI_Group *group2;
MPI_Group * group_out;
int *ierr;
{
  *ierr = MPI_Group_intersection( *group1, *group2, group_out );
}

void   mpi_group_rank_( group, rank, ierr)
MPI_Group *group;
int * rank;
int *ierr;
{
  *ierr = MPI_Group_rank( *group, rank );
}

void   mpi_group_range_excl_( group, n, ranges, newgroup, ierr )
MPI_Group *group;
int *n;
int ranges[][3];
MPI_Group * newgroup;
int *ierr;
{
  *ierr = MPI_Group_range_excl( *group, *n, ranges, newgroup );
}

void   mpi_group_range_incl_( group, n, ranges, newgroup, ierr )
MPI_Group *group;
int *n;
int ranges[][3];
MPI_Group * newgroup;
int *ierr;
{
  *ierr = MPI_Group_range_incl( *group, *n, ranges, newgroup );
}

void   mpi_group_size_( group, size, ierr )
MPI_Group *group;
int * size;
int *ierr;
{
  *ierr = MPI_Group_size( *group, size );
}

void   mpi_group_translate_ranks_( group_a, n, ranks_a, group_b, ranks_b, ierr)

MPI_Group *group_a;
int *n;
int * ranks_a;
MPI_Group *group_b;
int * ranks_b;
int *ierr;
{
  *ierr = MPI_Group_translate_ranks( *group_a, *n, ranks_a, *group_b, ranks_b );
}

void   mpi_group_union_( group1, group2, group_out, ierr )
MPI_Group *group1;
MPI_Group *group2;
MPI_Group * group_out;
int *ierr;
{
  *ierr = MPI_Group_union( *group1, *group2, group_out );
}

void   mpi_intercomm_create_( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr )
MPI_Comm *local_comm;
int *local_leader;
MPI_Comm *peer_comm;
int *remote_leader;
int *tag;
MPI_Comm * comm_out;
int *ierr;
{
  *ierr = MPI_Intercomm_create( *local_comm, *local_leader, *peer_comm, *remote_leader, *tag, comm_out );
}

void   mpi_intercomm_merge_( comm, high, comm_out, ierr )
MPI_Comm *comm;
int *high;
MPI_Comm * comm_out;
int *ierr;
{
  *ierr = MPI_Intercomm_merge( *comm, *high, comm_out );
}

void   mpi_keyval_create_( copy_fn, delete_fn, keyval, extra_state, ierr )
MPI_Copy_function * copy_fn;
MPI_Delete_function * delete_fn;
int * keyval;
void * extra_state;
int *ierr;
{
  *ierr = MPI_Keyval_create( copy_fn, delete_fn, keyval, extra_state );
}

void   mpi_keyval_free_( keyval, ierr )
int * keyval;
int *ierr;
{
  *ierr = MPI_Keyval_free( keyval );
}

void  mpi_abort_( comm, errorcode , ierr)
MPI_Comm *comm;
int *errorcode;
int *ierr;
{
  *ierr = MPI_Abort( *comm, *errorcode );
}

void  mpi_error_class_( errorcode, errorclass, ierr )
int *errorcode;
int * errorclass;
int *ierr;
{
  *ierr = MPI_Error_class( *errorcode, errorclass );
}

void  mpi_errhandler_create_( function, errhandler , ierr)
MPI_Handler_function * function;
MPI_Errhandler * errhandler;
int *ierr;
{
  *ierr = MPI_Errhandler_create( function, errhandler );
}

void  mpi_errhandler_free_( errhandler, ierr )
MPI_Errhandler * errhandler;
int *ierr;
{
  *ierr = MPI_Errhandler_free( errhandler );
}

void  mpi_errhandler_get_( comm, errhandler, ierr )
MPI_Comm *comm;
MPI_Errhandler * errhandler;
int *ierr;
{
  *ierr = MPI_Errhandler_get( *comm, errhandler );
}

void  mpi_error_string_( errorcode, string, resultlen, ierr )
int *errorcode;
char * string;
int * resultlen;
int *ierr;
{
  *ierr = MPI_Error_string( *errorcode, string, resultlen );
}

void  mpi_errhandler_set_( comm, errhandler, ierr )
MPI_Comm *comm;
MPI_Errhandler *errhandler;
int *ierr;
{
  *ierr = MPI_Errhandler_set( *comm, *errhandler );
}

void  mpi_finalize_( ierr )
int *ierr;
{
  *ierr = MPI_Finalize(  );
}

void  mpi_get_processor_name_( name, resultlen, ierr )
char * name;
int * resultlen;
int *ierr;
{
  *ierr = MPI_Get_processor_name( name, resultlen );
}

void  mpi_init_( )
{
  MPI_Init( 0, (char ***)0);
}

#ifdef TAU_MPI_THREADED
void  mpi_init_thread_ (required, provided, ierr )
int *required;
int *provided;
int *ierr;
{
  *ierr = MPI_Init_thread( 0, (char ***)0, *required, provided );
}
#endif /* TAU_MPI_THREADED */



/*
int  mpi_initialized_( flag, ierr )
int * flag;
int *ierr;
{
  *ierr = MPI_Initialized( flag );
}
*/

double  mpi_wtick_( )
{
  return MPI_Wtick(  );
}

double  mpi_wtime(  )
{
  return MPI_Wtime(  );
}

double  mpi_wtime_(  )
{
  return MPI_Wtime(  );
}

void  mpi_address_( location, address , ierr)
void * location;
MPI_Aint * address;
int *ierr;
{
  *ierr = MPI_Address( location, address );
}

void  mpi_bsend_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Bsend( buf, *count, *datatype, *dest, *tag, *comm );
}

void  mpi_bsend_init_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Bsend_init( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_buffer_attach_( buffer, size, ierr )
void * buffer;
int *size;
int *ierr;
{
  *ierr = MPI_Buffer_attach( buffer, *size );
}

void  mpi_buffer_detach_( buffer, size, ierr )
void * buffer;
int * size;
int *ierr;
{
  *ierr = MPI_Buffer_detach( buffer, size );
}

void  mpi_cancel_( request, ierr )
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Cancel( request );
}

void  mpi_request_free_( request, ierr )
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Request_free( request );
}

void  mpi_recv_init_( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *source;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Recv_init( buf, *count, *datatype, *source, *tag, *comm, request );
}

void  mpi_send_init_( buf, count, datatype, dest, tag, comm, request, ierr )	
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Send_init( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void   mpi_get_elements_( status, datatype, elements, ierr )
MPI_Status * status;
MPI_Datatype *datatype;
int * elements;
int *ierr;
{
  *ierr = MPI_Get_elements( status, *datatype, elements );
}

void  mpi_get_count_( status, datatype, count, ierr )
MPI_Status * status;
MPI_Datatype *datatype;
int * count;
int *ierr;
{
  *ierr = MPI_Get_count( status, *datatype, count );
}

void  mpi_ibsend_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Ibsend( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_iprobe_( source, tag, comm, flag, status, ierr )
int *source;
int *tag;
MPI_Comm *comm;
int * flag;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Iprobe( *source, *tag, *comm, flag, status );
}

void  mpi_irecv_( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *source;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Irecv( buf, *count, *datatype, *source, *tag, *comm, request );
}

void  mpi_irsend_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Irsend( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_isend_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Isend( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_issend_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Issend( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void   mpi_pack_( inbuf, incount, type, outbuf, outcount, position, comm, ierr )
void * inbuf;
int *incount;
MPI_Datatype *type;
void * outbuf;
int *outcount;
int * position;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Pack( inbuf, *incount, *type, outbuf, *outcount, position, *comm );
}

void   mpi_pack_size_( incount, datatype, comm, size, ierr )
int *incount;
MPI_Datatype *datatype;
MPI_Comm *comm;
int * size;
int *ierr;
{
  *ierr = MPI_Pack_size( *incount, *datatype, *comm, size );
}

void  mpi_probe_( source, tag, comm, status, ierr )
int *source;
int *tag;
MPI_Comm *comm;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Probe( *source, *tag, *comm, status );
}

void  mpi_recv_( buf, count, datatype, source, tag, comm, status, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *source;
int *tag;
MPI_Comm *comm;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Recv( buf, *count, *datatype, *source, *tag, *comm, status );
}

void  mpi_rsend_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Rsend( buf, *count, *datatype, *dest, *tag, *comm );
}

void  mpi_rsend_init_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Rsend_init( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_send_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Send( buf, *count, *datatype, *dest, *tag, *comm );
}

void  mpi_sendrecv_( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr )
void * sendbuf;
int *sendcount;
MPI_Datatype *sendtype;
int *dest;
int *sendtag;
void * recvbuf;
int *recvcount;
MPI_Datatype *recvtype;
int *source;
int *recvtag;
MPI_Comm *comm;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Sendrecv( sendbuf, *sendcount, *sendtype, *dest, *sendtag, recvbuf, *recvcount, *recvtype, *source, *recvtag, *comm, status );
}

void  mpi_sendrecv_replace_( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *sendtag;
int *source;
int *recvtag;
MPI_Comm *comm;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Sendrecv_replace( buf, *count, *datatype, *dest, *sendtag, *source, *recvtag, *comm, status );
}

void  mpi_ssend_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Ssend( buf, *count, *datatype, *dest, *tag, *comm );
}

void  mpi_ssend_init_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Ssend_init( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_start_( request, ierr )
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Start( request );
}

void  mpi_startall_( count, array_of_requests, ierr )
int *count;
MPI_Request * array_of_requests;
int *ierr;
{
  *ierr = MPI_Startall( *count, array_of_requests );
}

void   mpi_test_( request, flag, status, ierr )
MPI_Request * request;
int * flag;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Test( request, flag, status );
}

void  mpi_testall_( count, array_of_requests, flag, array_of_statuses, ierr )
int *count;
MPI_Request * array_of_requests;
int * flag;
MPI_Status * array_of_statuses;
int *ierr;
{
  *ierr = MPI_Testall( *count, array_of_requests, flag, array_of_statuses );
}

void  mpi_testany_( count, array_of_requests, index, flag, status, ierr )
int *count;
MPI_Request * array_of_requests;
int * index;
int * flag;
MPI_Status * status;
int *ierr;
{
  *ierr	 = MPI_Testany( *count, array_of_requests, index, flag, status );
}

void  mpi_test_cancelled_( status, flag, ierr )
MPI_Status * status;
int * flag;
int *ierr;
{
  *ierr = MPI_Test_cancelled( status, flag );
}

void  mpi_testsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )
int *incount;
MPI_Request * array_of_requests;
int * outcount;
int * array_of_indices;
MPI_Status * array_of_statuses;
int *ierr;
{
  *ierr = MPI_Testsome( *incount, array_of_requests, outcount, array_of_indices, array_of_statuses );
}

void   mpi_type_commit_( datatype, ierr )
MPI_Datatype * datatype;
int *ierr;
{
  *ierr = MPI_Type_commit( datatype );
}

void  mpi_type_contiguous_( count, old_type, newtype, ierr )
int *count;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_contiguous( *count, *old_type, newtype );
}

void  mpi_type_extent_( datatype, extent, ierr )
MPI_Datatype *datatype;
MPI_Aint * extent;
int *ierr;
{
  *ierr = MPI_Type_extent( *datatype, extent );
}

void   mpi_type_free_( datatype, ierr )
MPI_Datatype * datatype;
int *ierr;
{
  *ierr = MPI_Type_free( datatype );
}

void  mpi_type_hindexed_( count, blocklens, indices, old_type, newtype, ierr )
int *count;
int * blocklens;
MPI_Aint * indices;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_hindexed( *count, blocklens, indices, *old_type, newtype );
}

void  mpi_type_hvector_( count, blocklen, stride, old_type, newtype, ierr )
int *count;
int *blocklen;
MPI_Aint *stride;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_hvector( *count, *blocklen, *stride, *old_type, newtype );
}

void  mpi_type_indexed_( count, blocklens, indices, old_type, newtype, ierr )
int *count;
int * blocklens;
int * indices;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_indexed( *count, blocklens, indices, *old_type, newtype );
}

void   mpi_type_lb_( datatype, displacement, ierr )
MPI_Datatype *datatype;
MPI_Aint * displacement;
int *ierr;
{
  *ierr = MPI_Type_lb( *datatype, displacement );
}

void   mpi_type_size_( datatype, size, ierr )
MPI_Datatype *datatype;
int * size;
int *ierr;
{
  *ierr = MPI_Type_size( *datatype, size );
}

void  mpi_type_struct_( count, blocklens, indices, old_types, newtype, ierr )
int *count;
int * blocklens;
MPI_Aint * indices;
MPI_Datatype * old_types;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_struct( *count, blocklens, indices, old_types, newtype );
}

void   mpi_type_ub_( datatype, displacement, ierr )
MPI_Datatype *datatype;
MPI_Aint * displacement;
int *ierr;
{
  *ierr = MPI_Type_ub( *datatype, displacement );
}

void  mpi_type_vector_( count, blocklen, stride, old_type, newtype, ierr )
int *count;
int *blocklen;
int *stride;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_vector( *count, *blocklen, *stride, *old_type, newtype );
}

void   mpi_unpack_( inbuf, insize, position, outbuf, outcount, type, comm, ierr )
void * inbuf;
int *insize;
int * position;
void * outbuf;
int *outcount;
MPI_Datatype *type;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Unpack( inbuf, *insize, position, outbuf, *outcount, *type, *comm );
}

void   mpi_wait_( request, status, ierr )
MPI_Request * request;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Wait( request, status );
}

void  mpi_waitall_( count, array_of_requests, array_of_statuses, ierr )
int *count;
MPI_Request * array_of_requests;
MPI_Status * array_of_statuses;
int *ierr;
{
  *ierr = MPI_Waitall( *count, array_of_requests, array_of_statuses );
}

void  mpi_waitany_( count, array_of_requests, index, status, ierr )
int *count;
MPI_Request * array_of_requests;
int * index;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Waitany( *count, array_of_requests, index, status );
}

void  mpi_waitsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )

int *incount;
MPI_Request * array_of_requests;
int * outcount;
int * array_of_indices;
MPI_Status * array_of_statuses;
int *ierr;
{
  *ierr = MPI_Waitsome( *incount, array_of_requests, outcount, array_of_indices, array_of_statuses );
}

void   mpi_cart_coords_( comm, rank, maxdims, coords, ierr )
MPI_Comm *comm;
int *rank;
int *maxdims;
int * coords;
int *ierr;
{
  *ierr = MPI_Cart_coords( *comm, *rank, *maxdims, coords );
}

void   mpi_cart_create_( comm_old, ndims, dims, periods, reorder, comm_cart, ierr )
MPI_Comm *comm_old;
int *ndims;
int * dims;
int * periods;
int *reorder;
MPI_Comm * comm_cart;
int *ierr;
{
  *ierr = MPI_Cart_create( *comm_old, *ndims, dims, periods, *reorder, comm_cart );
}

void   mpi_cart_get_( comm, maxdims, dims, periods, coords, ierr )
MPI_Comm *comm;
int *maxdims;
int * dims;
int * periods;
int * coords;
int *ierr;
{
  *ierr = MPI_Cart_get( *comm, *maxdims, dims, periods, coords );
}

void   mpi_cart_map_( comm_old, ndims, dims, periods, newrank, ierr )
MPI_Comm *comm_old;
int *ndims;
int * dims;
int * periods;
int * newrank;
int *ierr;
{
  *ierr = MPI_Cart_map( *comm_old, *ndims, dims, periods, newrank );
}

void   mpi_cart_rank_( comm, coords, rank, ierr )
MPI_Comm *comm;
int * coords;
int * rank;
int *ierr;
{
  *ierr = MPI_Cart_rank( *comm, coords, rank );
}

void   mpi_cart_shift_( comm, direction, displ, source, dest, ierr )
MPI_Comm *comm;
int *direction;
int *displ;
int * source;
int * dest;
int *ierr;
{
  *ierr = MPI_Cart_shift( *comm, *direction, *displ, source, dest );
}

void   mpi_cart_sub_( comm, remain_dims, comm_new, ierr )
MPI_Comm *comm;
int * remain_dims;
MPI_Comm * comm_new;
int *ierr;
{
  *ierr = MPI_Cart_sub( *comm, remain_dims, comm_new );
}

void   mpi_cartdim_get_( comm, ndims, ierr )
MPI_Comm *comm;
int * ndims;
int *ierr;
{
  *ierr = MPI_Cartdim_get( *comm, ndims );
}

void  mpi_dims_create_( nnodes, ndims, dims, ierr )
int *nnodes;
int *ndims;
int * dims;
int *ierr;
{
  *ierr = MPI_Dims_create( *nnodes, *ndims, dims );
}

void   mpi_graph_create_( comm_old, nnodes, index, edges, reorder, comm_graph, ierr )
MPI_Comm *comm_old;
int *nnodes;
int * index;
int * edges;
int *reorder;
MPI_Comm * comm_graph;
int *ierr;
{
  *ierr = MPI_Graph_create( *comm_old, *nnodes, index, edges, *reorder, comm_graph );
}

void   mpi_graph_get_( comm, maxindex, maxedges, index, edges, ierr )
MPI_Comm *comm;
int *maxindex;
int *maxedges;
int * index;
int * edges;
int *ierr;
{
  *ierr = MPI_Graph_get( *comm, *maxindex, *maxedges, index, edges );
}

void   mpi_graph_map_( comm_old, nnodes, index, edges, newrank, ierr )
MPI_Comm *comm_old;
int *nnodes;
int * index;
int * edges;
int * newrank;
int *ierr;
{
  *ierr = MPI_Graph_map( *comm_old, *nnodes, index, edges, newrank );
}

void   mpi_graph_neighbors_( comm, rank, maxneighbors, neighbors, ierr )
MPI_Comm *comm;
int *rank;
int  *maxneighbors;
int * neighbors;
int *ierr;
{
  *ierr = MPI_Graph_neighbors( *comm, *rank, *maxneighbors, neighbors );
}

void   mpi_graph_neighbors_count_( comm, rank, nneighbors, ierr )
MPI_Comm *comm;
int *rank;
int * nneighbors;
int *ierr;
{
  *ierr = MPI_Graph_neighbors_count( *comm, *rank, nneighbors );
}

void   mpi_graphdims_get_( comm, nnodes, nedges, ierr )
MPI_Comm *comm;
int * nnodes;
int * nedges;
int *ierr;
{
  *ierr = MPI_Graphdims_get( *comm, nnodes, nedges );
}

void   mpi_topo_test_( comm, top_type, ierr )
MPI_Comm *comm;
int * top_type;
int *ierr;
{
  *ierr = MPI_Topo_test( *comm, top_type );
}

/* lowercase interface without underscore begins here */

void  mpi_allgather( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr)
void * sendbuf;
int *sendcount;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcount;
MPI_Datatype *recvtype;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Allgather( sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, *recvtype, *comm );

}

void  mpi_allgatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr)
void * sendbuf;
int *sendcount;
MPI_Datatype *sendtype;
void * recvbuf;
int * recvcounts;
int * displs;
MPI_Datatype *recvtype;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Allgatherv( sendbuf, *sendcount, *sendtype, recvbuf, recvcounts, displs, *recvtype, *comm );

}

void   mpi_allreduce( sendbuf, recvbuf, count, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
int *count;
MPI_Datatype *datatype;
MPI_Op *op;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Allreduce( sendbuf, recvbuf, *count, *datatype, *op, *comm );

}

void  mpi_alltoall( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr)
void * sendbuf;
int *sendcount;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcnt;
MPI_Datatype *recvtype;
MPI_Comm *comm;
int *ierr; 
{
  *ierr = MPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcnt, *recvtype, *comm );
}

void   mpi_alltoallv( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr)
void * sendbuf;
int * sendcnts;
int * sdispls;
MPI_Datatype *sendtype;
void * recvbuf;
int * recvcnts;
int * rdispls;
MPI_Datatype *recvtype;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Alltoallv( sendbuf, sendcnts, sdispls, *sendtype, recvbuf, recvcnts, rdispls, *recvtype, *comm );

}

void   mpi_barrier( comm , ierr)
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Barrier( *comm );
}

void   mpi_bcast( buffer, count, datatype, root, comm , ierr)
void * buffer;
int *count;
MPI_Datatype *datatype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Bcast( buffer, *count, *datatype, *root, *comm );

}

void  mpi_gather( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr)
void * sendbuf;
int *sendcnt;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcount;
MPI_Datatype *recvtype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  
  *ierr = MPI_Gather( sendbuf, *sendcnt, *sendtype, recvbuf, *recvcount, *recvtype, *root, *comm );

}

void mpi_gatherv( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr)
void * sendbuf;
int *sendcnt;
MPI_Datatype *sendtype;
void * recvbuf;
int * recvcnts;
int * displs;
MPI_Datatype *recvtype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Gatherv( sendbuf, *sendcnt, *sendtype, recvbuf, recvcnts, displs, *recvtype, *root, *comm );

}

void mpi_op_create( function, commute, op , ierr)
MPI_User_function * function;
int *commute;
MPI_Op * op;
int *ierr;
{
  *ierr = MPI_Op_create( function, *commute, op );

}

void  mpi_op_free( op , ierr)
MPI_Op * op;
int *ierr;
{
  *ierr = MPI_Op_free( op );
}

void mpi_reduce_scatter( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
int * recvcnts;
MPI_Datatype *datatype;
MPI_Op *op;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Reduce_scatter( sendbuf, recvbuf, recvcnts, *datatype, *op, *comm );
}

void mpi_reduce( sendbuf, recvbuf, count, datatype, op, root, comm , ierr)
void * sendbuf;
void * recvbuf;
int *count;
MPI_Datatype *datatype;
MPI_Op *op;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Reduce( sendbuf, recvbuf, *count, *datatype, *op, *root, *comm );
}

void mpi_scan( sendbuf, recvbuf, count, datatype, op, comm, ierr)
void * sendbuf;
void * recvbuf;
int *count;
MPI_Datatype *datatype;
MPI_Op *op;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Scan( sendbuf, recvbuf, *count, *datatype, *op, *comm );
}

void   mpi_scatter( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr )
void * sendbuf;
int *sendcnt;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcnt;
MPI_Datatype *recvtype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Scatter( sendbuf, *sendcnt, *sendtype, recvbuf, *recvcnt, *recvtype, *root, *comm );
}

void  mpi_scatterv( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr)
void * sendbuf;
int * sendcnts;
int * displs;
MPI_Datatype *sendtype;
void * recvbuf;
int *recvcnt;
MPI_Datatype *recvtype;
int *root;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Scatterv( sendbuf, sendcnts, displs, *sendtype, recvbuf, *recvcnt, *recvtype, *root, *comm );
}

void   mpi_attr_delete( comm, keyval, ierr)
MPI_Comm *comm;
int *keyval;
int *ierr;
{
  *ierr = MPI_Attr_delete( *comm, *keyval );
}

void mpi_attr_get( comm, keyval, attr_value, flag , ierr)
MPI_Comm *comm;
int *keyval;
void * attr_value;
int * flag;
int *ierr;
{
  *ierr = MPI_Attr_get( *comm, *keyval, attr_value, flag );
}

void   mpi_attr_put( comm, keyval, attr_value, ierr)
MPI_Comm *comm;
int *keyval;
void * attr_value;
int *ierr;
{
  *ierr = MPI_Attr_put( *comm, *keyval, attr_value );
}

void  mpi_comm_compare( comm1, comm2, result, ierr )
MPI_Comm *comm1;
MPI_Comm *comm2;
int * result;
int *ierr;
{
  *ierr = MPI_Comm_compare( *comm1, *comm2, result );
}

void  mpi_comm_create( comm, group, comm_out, ierr )
MPI_Comm *comm;
MPI_Group *group;
MPI_Comm * comm_out;
int *ierr;
{
  *ierr = MPI_Comm_create( *comm, *group, comm_out );
}

void   mpi_comm_dup( comm, comm_out, ierr )
MPI_Comm *comm;
MPI_Comm * comm_out;
int *ierr;
{
  *ierr = MPI_Comm_dup( *comm, comm_out );
}

void   mpi_comm_free( comm, ierr)
MPI_Comm * comm;
int *ierr;
{
  *ierr = MPI_Comm_free( comm );
}

void   mpi_comm_group( comm, group, ierr )
MPI_Comm *comm;
MPI_Group * group;
int *ierr;
{
  *ierr = MPI_Comm_group( *comm, group );
}

void   mpi_comm_rank( comm, rank, ierr )
MPI_Comm *comm;
int * rank;
int *ierr;
{
  *ierr = MPI_Comm_rank( *comm, rank );
}

void   mpi_comm_remote_group( comm, group, ierr )
MPI_Comm *comm;
MPI_Group * group;
int *ierr;
{
  *ierr = MPI_Comm_remote_group( *comm, group );
}

void   mpi_comm_remote_size( comm, size, ierr )
MPI_Comm *comm;
int * size;
int *ierr;
{
  *ierr = MPI_Comm_remote_size( *comm, size );
}

void   mpi_comm_size( comm, size , ierr)
MPI_Comm *comm;
int * size;
int *ierr;
{
  *ierr = MPI_Comm_size( *comm, size );
}

void   mpi_comm_split( comm, color, key, comm_out, ierr )
MPI_Comm *comm;
int *color;
int *key;
MPI_Comm * comm_out;
int *ierr;
{
  MPI_Comm l_comm_out;
  *ierr = MPI_Comm_split( *comm, *color, *key, &l_comm_out );
  *comm_out = l_comm_out; 
}

void   mpi_comm_test_inter( comm, flag, ierr )
MPI_Comm *comm;
int * flag;
int *ierr;
{
  *ierr = MPI_Comm_test_inter( *comm, flag );
}

void   mpi_group_compare( group1, group2, result, ierr )
MPI_Group *group1;
MPI_Group *group2;
int * result;
int *ierr;
{
  *ierr = MPI_Group_compare( *group1, *group2, result );
}

void   mpi_group_difference( group1, group2, group_out, ierr )
MPI_Group *group1;
MPI_Group *group2;
MPI_Group * group_out;
int *ierr;
{
  *ierr = MPI_Group_difference( *group1, *group2, group_out );
}

void   mpi_group_excl( group, n, ranks, newgroup, ierr )
MPI_Group *group;
int *n;
int * ranks;
MPI_Group * newgroup;
int *ierr;
{
  *ierr = MPI_Group_excl( *group, *n, ranks, newgroup );
}

void   mpi_group_free( group, ierr)
MPI_Group * group;
int *ierr;
{
  *ierr = MPI_Group_free( group );
}

void   mpi_group_incl( group, n, ranks, group_out, ierr )
MPI_Group *group;
int *n;
int * ranks;
MPI_Group * group_out;
int *ierr;
{
  *ierr = MPI_Group_incl( *group, *n, ranks, group_out );
}

void   mpi_group_intersection( group1, group2, group_out, ierr )
MPI_Group *group1;
MPI_Group *group2;
MPI_Group * group_out;
int *ierr;
{
  *ierr = MPI_Group_intersection( *group1, *group2, group_out );
}

void   mpi_group_rank( group, rank, ierr)
MPI_Group *group;
int * rank;
int *ierr;
{
  *ierr = MPI_Group_rank( *group, rank );
}

void   mpi_group_range_excl( group, n, ranges, newgroup, ierr )
MPI_Group *group;
int *n;
int ranges[][3];
MPI_Group * newgroup;
int *ierr;
{
  *ierr = MPI_Group_range_excl( *group, *n, ranges, newgroup );
}

void   mpi_group_range_incl( group, n, ranges, newgroup, ierr )
MPI_Group *group;
int *n;
int ranges[][3];
MPI_Group * newgroup;
int *ierr;
{
  *ierr = MPI_Group_range_incl( *group, *n, ranges, newgroup );
}

void   mpi_group_size( group, size, ierr )
MPI_Group *group;
int * size;
int *ierr;
{
  *ierr = MPI_Group_size( *group, size );
}

void   mpi_group_translate_ranks( group_a, n, ranks_a, group_b, ranks_b, ierr)

MPI_Group *group_a;
int *n;
int * ranks_a;
MPI_Group *group_b;
int * ranks_b;
int *ierr;
{
  *ierr = MPI_Group_translate_ranks( *group_a, *n, ranks_a, *group_b, ranks_b );
}

void   mpi_group_union( group1, group2, group_out, ierr )
MPI_Group *group1;
MPI_Group *group2;
MPI_Group * group_out;
int *ierr;
{
  *ierr = MPI_Group_union( *group1, *group2, group_out );
}

void   mpi_intercomm_create( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr )
MPI_Comm *local_comm;
int *local_leader;
MPI_Comm *peer_comm;
int *remote_leader;
int *tag;
MPI_Comm * comm_out;
int *ierr;
{
  *ierr = MPI_Intercomm_create( *local_comm, *local_leader, *peer_comm, *remote_leader, *tag, comm_out );
}

void   mpi_intercomm_merge( comm, high, comm_out, ierr )
MPI_Comm *comm;
int *high;
MPI_Comm * comm_out;
int *ierr;
{
  *ierr = MPI_Intercomm_merge( *comm, *high, comm_out );
}

void   mpi_keyval_create( copy_fn, delete_fn, keyval, extra_state, ierr )
MPI_Copy_function * copy_fn;
MPI_Delete_function * delete_fn;
int * keyval;
void * extra_state;
int *ierr;
{
  *ierr = MPI_Keyval_create( copy_fn, delete_fn, keyval, extra_state );
}

void   mpi_keyval_free( keyval, ierr )
int * keyval;
int *ierr;
{
  *ierr = MPI_Keyval_free( keyval );
}

void  mpi_abort( comm, errorcode , ierr)
MPI_Comm *comm;
int *errorcode;
int *ierr;
{
  *ierr = MPI_Abort( *comm, *errorcode );
}

void  mpi_error_class( errorcode, errorclass, ierr )
int *errorcode;
int * errorclass;
int *ierr;
{
  *ierr = MPI_Error_class( *errorcode, errorclass );
}

void  mpi_errhandler_create( function, errhandler , ierr)
MPI_Handler_function * function;
MPI_Errhandler * errhandler;
int *ierr;
{
  *ierr = MPI_Errhandler_create( function, errhandler );
}

void  mpi_errhandler_free( errhandler, ierr )
MPI_Errhandler * errhandler;
int *ierr;
{
  *ierr = MPI_Errhandler_free( errhandler );
}

void  mpi_errhandler_get( comm, errhandler, ierr )
MPI_Comm *comm;
MPI_Errhandler * errhandler;
int *ierr;
{
  *ierr = MPI_Errhandler_get( *comm, errhandler );
}

void  mpi_error_string( errorcode, string, resultlen, ierr )
int *errorcode;
char * string;
int * resultlen;
int *ierr;
{
  *ierr = MPI_Error_string( *errorcode, string, resultlen );
}

void  mpi_errhandler_set( comm, errhandler, ierr )
MPI_Comm *comm;
MPI_Errhandler *errhandler;
int *ierr;
{
  *ierr = MPI_Errhandler_set( *comm, *errhandler );
}

void  mpi_finalize( ierr )
int *ierr;
{
  *ierr = MPI_Finalize(  );
}

void  mpi_get_processor_name( name, resultlen, ierr )
char * name;
int * resultlen;
int *ierr;
{
  *ierr = MPI_Get_processor_name( name, resultlen );
}

void  mpi_init( )
{
  MPI_Init( 0, (char ***)0);
}

#ifdef TAU_MPI_THREADED
void  mpi_init_thread(required, provided, ierr )
int *required;
int *provided;
int *ierr;
{
  *ierr = MPI_Init_thread( 0, (char ***)0, *required, provided );
}
#endif /* TAU_MPI_THREADED */



/*
int  mpi_initialized( flag, ierr )
int * flag;
int *ierr;
{
  *ierr = MPI_Initialized( flag );
}
*/

void  mpi_address( location, address , ierr)
void * location;
MPI_Aint * address;
int *ierr;
{
  *ierr = MPI_Address( location, address );
}

void  mpi_bsend( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Bsend( buf, *count, *datatype, *dest, *tag, *comm );
}

void  mpi_bsend_init( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Bsend_init( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_buffer_attach( buffer, size, ierr )
void * buffer;
int *size;
int *ierr;
{
  *ierr = MPI_Buffer_attach( buffer, *size );
}

void  mpi_buffer_detach( buffer, size, ierr )
void * buffer;
int * size;
int *ierr;
{
  *ierr = MPI_Buffer_detach( buffer, size );
}

void  mpi_cancel( request, ierr )
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Cancel( request );
}

void  mpi_request_free( request, ierr )
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Request_free( request );
}

void  mpi_recv_init( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *source;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Recv_init( buf, *count, *datatype, *source, *tag, *comm, request );
}

void  mpi_send_init( buf, count, datatype, dest, tag, comm, request, ierr )	
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Send_init( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void   mpi_get_elements( status, datatype, elements, ierr )
MPI_Status * status;
MPI_Datatype *datatype;
int * elements;
int *ierr;
{
  *ierr = MPI_Get_elements( status, *datatype, elements );
}

void  mpi_get_count( status, datatype, count, ierr )
MPI_Status * status;
MPI_Datatype *datatype;
int * count;
int *ierr;
{
  *ierr = MPI_Get_count( status, *datatype, count );
}

void  mpi_ibsend( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Ibsend( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_iprobe( source, tag, comm, flag, status, ierr )
int *source;
int *tag;
MPI_Comm *comm;
int * flag;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Iprobe( *source, *tag, *comm, flag, status );
}

void  mpi_irecv( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *source;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Irecv( buf, *count, *datatype, *source, *tag, *comm, request );
}

void  mpi_irsend( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Irsend( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_isend( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Isend( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_issend( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Issend( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void   mpi_pack( inbuf, incount, type, outbuf, outcount, position, comm, ierr )
void * inbuf;
int *incount;
MPI_Datatype *type;
void * outbuf;
int *outcount;
int * position;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Pack( inbuf, *incount, *type, outbuf, *outcount, position, *comm );
}

void   mpi_pack_size( incount, datatype, comm, size, ierr )
int *incount;
MPI_Datatype *datatype;
MPI_Comm *comm;
int * size;
int *ierr;
{
  *ierr = MPI_Pack_size( *incount, *datatype, *comm, size );
}

void  mpi_probe( source, tag, comm, status, ierr )
int *source;
int *tag;
MPI_Comm *comm;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Probe( *source, *tag, *comm, status );
}

void  mpi_recv( buf, count, datatype, source, tag, comm, status, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *source;
int *tag;
MPI_Comm *comm;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Recv( buf, *count, *datatype, *source, *tag, *comm, status );
}

void  mpi_rsend( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Rsend( buf, *count, *datatype, *dest, *tag, *comm );
}

void  mpi_rsend_init( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Rsend_init( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_send( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Send( buf, *count, *datatype, *dest, *tag, *comm );
}

void  mpi_sendrecv( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr )
void * sendbuf;
int *sendcount;
MPI_Datatype *sendtype;
int *dest;
int *sendtag;
void * recvbuf;
int *recvcount;
MPI_Datatype *recvtype;
int *source;
int *recvtag;
MPI_Comm *comm;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Sendrecv( sendbuf, *sendcount, *sendtype, *dest, *sendtag, recvbuf, *recvcount, *recvtype, *source, *recvtag, *comm, status );
}

void  mpi_sendrecv_replace( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *sendtag;
int *source;
int *recvtag;
MPI_Comm *comm;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Sendrecv_replace( buf, *count, *datatype, *dest, *sendtag, *source, *recvtag, *comm, status );
}

void  mpi_ssend( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Ssend( buf, *count, *datatype, *dest, *tag, *comm );
}

void  mpi_ssend_init( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
int *count;
MPI_Datatype *datatype;
int *dest;
int *tag;
MPI_Comm *comm;
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Ssend_init( buf, *count, *datatype, *dest, *tag, *comm, request );
}

void  mpi_start( request, ierr )
MPI_Request * request;
int *ierr;
{
  *ierr = MPI_Start( request );
}

void  mpi_startall( count, array_of_requests, ierr )
int *count;
MPI_Request * array_of_requests;
int *ierr;
{
  *ierr = MPI_Startall( *count, array_of_requests );
}

void   mpi_test( request, flag, status, ierr )
MPI_Request * request;
int * flag;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Test( request, flag, status );
}

void  mpi_testall( count, array_of_requests, flag, array_of_statuses, ierr )
int *count;
MPI_Request * array_of_requests;
int * flag;
MPI_Status * array_of_statuses;
int *ierr;
{
  *ierr = MPI_Testall( *count, array_of_requests, flag, array_of_statuses );
}

void  mpi_testany( count, array_of_requests, index, flag, status, ierr )
int *count;
MPI_Request * array_of_requests;
int * index;
int * flag;
MPI_Status * status;
int *ierr;
{
  *ierr	 = MPI_Testany( *count, array_of_requests, index, flag, status );
}

void  mpi_test_cancelled( status, flag, ierr )
MPI_Status * status;
int * flag;
int *ierr;
{
  *ierr = MPI_Test_cancelled( status, flag );
}

void  mpi_testsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )
int *incount;
MPI_Request * array_of_requests;
int * outcount;
int * array_of_indices;
MPI_Status * array_of_statuses;
int *ierr;
{
  *ierr = MPI_Testsome( *incount, array_of_requests, outcount, array_of_indices, array_of_statuses );
}

void   mpi_type_commit( datatype, ierr )
MPI_Datatype * datatype;
int *ierr;
{
  *ierr = MPI_Type_commit( datatype );
}

void  mpi_type_contiguous( count, old_type, newtype, ierr )
int *count;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_contiguous( *count, *old_type, newtype );
}

void  mpi_type_extent( datatype, extent, ierr )
MPI_Datatype *datatype;
MPI_Aint * extent;
int *ierr;
{
  *ierr = MPI_Type_extent( *datatype, extent );
}

void   mpi_type_free( datatype, ierr )
MPI_Datatype * datatype;
int *ierr;
{
  *ierr = MPI_Type_free( datatype );
}

void  mpi_type_hindexed( count, blocklens, indices, old_type, newtype, ierr )
int *count;
int * blocklens;
MPI_Aint * indices;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_hindexed( *count, blocklens, indices, *old_type, newtype );
}

void  mpi_type_hvector( count, blocklen, stride, old_type, newtype, ierr )
int *count;
int *blocklen;
MPI_Aint *stride;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_hvector( *count, *blocklen, *stride, *old_type, newtype );
}

void  mpi_type_indexed( count, blocklens, indices, old_type, newtype, ierr )
int *count;
int * blocklens;
int * indices;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_indexed( *count, blocklens, indices, *old_type, newtype );
}

void   mpi_type_lb( datatype, displacement, ierr )
MPI_Datatype *datatype;
MPI_Aint * displacement;
int *ierr;
{
  *ierr = MPI_Type_lb( *datatype, displacement );
}

void   mpi_type_size( datatype, size, ierr )
MPI_Datatype *datatype;
int * size;
int *ierr;
{
  *ierr = MPI_Type_size( *datatype, size );
}

void  mpi_type_struct( count, blocklens, indices, old_types, newtype, ierr )
int *count;
int * blocklens;
MPI_Aint * indices;
MPI_Datatype * old_types;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_struct( *count, blocklens, indices, old_types, newtype );
}

void   mpi_type_ub( datatype, displacement, ierr )
MPI_Datatype *datatype;
MPI_Aint * displacement;
int *ierr;
{
  *ierr = MPI_Type_ub( *datatype, displacement );
}

void  mpi_type_vector( count, blocklen, stride, old_type, newtype, ierr )
int *count;
int *blocklen;
int *stride;
MPI_Datatype *old_type;
MPI_Datatype * newtype;
int *ierr;
{
  *ierr = MPI_Type_vector( *count, *blocklen, *stride, *old_type, newtype );
}

void   mpi_unpack( inbuf, insize, position, outbuf, outcount, type, comm, ierr )
void * inbuf;
int *insize;
int * position;
void * outbuf;
int *outcount;
MPI_Datatype *type;
MPI_Comm *comm;
int *ierr;
{
  *ierr = MPI_Unpack( inbuf, *insize, position, outbuf, *outcount, *type, *comm );
}

void   mpi_wait( request, status, ierr )
MPI_Request * request;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Wait( request, status );
}

void  mpi_waitall( count, array_of_requests, array_of_statuses, ierr )
int *count;
MPI_Request * array_of_requests;
MPI_Status * array_of_statuses;
int *ierr;
{
  *ierr = MPI_Waitall( *count, array_of_requests, array_of_statuses );
}

void  mpi_waitany( count, array_of_requests, index, status, ierr )
int *count;
MPI_Request * array_of_requests;
int * index;
MPI_Status * status;
int *ierr;
{
  *ierr = MPI_Waitany( *count, array_of_requests, index, status );
}

void  mpi_waitsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )

int *incount;
MPI_Request * array_of_requests;
int * outcount;
int * array_of_indices;
MPI_Status * array_of_statuses;
int *ierr;
{
  *ierr = MPI_Waitsome( *incount, array_of_requests, outcount, array_of_indices, array_of_statuses );
}

void   mpi_cart_coords( comm, rank, maxdims, coords, ierr )
MPI_Comm *comm;
int *rank;
int *maxdims;
int * coords;
int *ierr;
{
  *ierr = MPI_Cart_coords( *comm, *rank, *maxdims, coords );
}

void   mpi_cart_create( comm_old, ndims, dims, periods, reorder, comm_cart, ierr )
MPI_Comm *comm_old;
int *ndims;
int * dims;
int * periods;
int *reorder;
MPI_Comm * comm_cart;
int *ierr;
{
  *ierr = MPI_Cart_create( *comm_old, *ndims, dims, periods, *reorder, comm_cart );
}

void   mpi_cart_get( comm, maxdims, dims, periods, coords, ierr )
MPI_Comm *comm;
int *maxdims;
int * dims;
int * periods;
int * coords;
int *ierr;
{
  *ierr = MPI_Cart_get( *comm, *maxdims, dims, periods, coords );
}

void   mpi_cart_map( comm_old, ndims, dims, periods, newrank, ierr )
MPI_Comm *comm_old;
int *ndims;
int * dims;
int * periods;
int * newrank;
int *ierr;
{
  *ierr = MPI_Cart_map( *comm_old, *ndims, dims, periods, newrank );
}

void   mpi_cart_rank( comm, coords, rank, ierr )
MPI_Comm *comm;
int * coords;
int * rank;
int *ierr;
{
  *ierr = MPI_Cart_rank( *comm, coords, rank );
}

void   mpi_cart_shift( comm, direction, displ, source, dest, ierr )
MPI_Comm *comm;
int *direction;
int *displ;
int * source;
int * dest;
int *ierr;
{
  *ierr = MPI_Cart_shift( *comm, *direction, *displ, source, dest );
}

void   mpi_cart_sub( comm, remain_dims, comm_new, ierr )
MPI_Comm *comm;
int * remain_dims;
MPI_Comm * comm_new;
int *ierr;
{
  *ierr = MPI_Cart_sub( *comm, remain_dims, comm_new );
}

void   mpi_cartdim_get( comm, ndims, ierr )
MPI_Comm *comm;
int * ndims;
int *ierr;
{
  *ierr = MPI_Cartdim_get( *comm, ndims );
}

void  mpi_dims_create( nnodes, ndims, dims, ierr )
int *nnodes;
int *ndims;
int * dims;
int *ierr;
{
  *ierr = MPI_Dims_create( *nnodes, *ndims, dims );
}

void   mpi_graph_create( comm_old, nnodes, index, edges, reorder, comm_graph, ierr )
MPI_Comm *comm_old;
int *nnodes;
int * index;
int * edges;
int *reorder;
MPI_Comm * comm_graph;
int *ierr;
{
  *ierr = MPI_Graph_create( *comm_old, *nnodes, index, edges, *reorder, comm_graph );
}

void   mpi_graph_get( comm, maxindex, maxedges, index, edges, ierr )
MPI_Comm *comm;
int *maxindex;
int *maxedges;
int * index;
int * edges;
int *ierr;
{
  *ierr = MPI_Graph_get( *comm, *maxindex, *maxedges, index, edges );
}

void   mpi_graph_map( comm_old, nnodes, index, edges, newrank, ierr )
MPI_Comm *comm_old;
int *nnodes;
int * index;
int * edges;
int * newrank;
int *ierr;
{
  *ierr = MPI_Graph_map( *comm_old, *nnodes, index, edges, newrank );
}

void   mpi_graph_neighbors( comm, rank, maxneighbors, neighbors, ierr )
MPI_Comm *comm;
int *rank;
int  *maxneighbors;
int * neighbors;
int *ierr;
{
  *ierr = MPI_Graph_neighbors( *comm, *rank, *maxneighbors, neighbors );
}

void   mpi_graph_neighbors_count( comm, rank, nneighbors, ierr )
MPI_Comm *comm;
int *rank;
int * nneighbors;
int *ierr;
{
  *ierr = MPI_Graph_neighbors_count( *comm, *rank, nneighbors );
}

void   mpi_graphdims_get( comm, nnodes, nedges, ierr )
MPI_Comm *comm;
int * nnodes;
int * nedges;
int *ierr;
{
  *ierr = MPI_Graphdims_get( *comm, nnodes, nedges );
}

void   mpi_topo_test( comm, top_type, ierr )
MPI_Comm *comm;
int * top_type;
int *ierr;
{
  *ierr = MPI_Topo_test( *comm, top_type );
}


