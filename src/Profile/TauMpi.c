#include <Profile/Profiler.h>
#include <stdio.h>
#include <mpi.h>


/* Wrappers that call TAU routines */

int  MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  int return_value;
  TAU_PROFILE_TIMER(tautimer, "MPI_Send()", " ", TAU_MESSAGE);
 

  TAU_TRACE_SENDMSG(tag, dest, count);

  TAU_PROFILE_START(tautimer);

  /* Actual MPI Call */
  return_value = PMPI_Send(buf, count, datatype, dest, tag, comm);

  TAU_PROFILE_STOP(tautimer);
  return return_value;
}


int MPI_Recv( void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status )
{
  int recv_count;
  int return_value;
  MPI_Status stat;



  TAU_PROFILE_TIMER(tautimer, "MPI_Recv()", " ", TAU_MESSAGE);

  TAU_PROFILE_START(tautimer);


  /* Actual MPI Call */
  return_value = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
 
  TAU_PROFILE_STOP(tautimer);

  MPI_Get_count(status,  datatype, &recv_count);
  TAU_TRACE_RECVMSG((*status).MPI_TAG, (*status).MPI_SOURCE, recv_count);

  return return_value;
}

/* The following is automatically generated using MPICH wrappergen */


int   MPI_Allgather( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm )
void * sendbuf;
int sendcount;
MPI_Datatype sendtype;
void * recvbuf;
int recvcount;
MPI_Datatype recvtype;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allgather()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Allgather( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Allgatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm )
void * sendbuf;
int sendcount;
MPI_Datatype sendtype;
void * recvbuf;
int * recvcounts;
int * displs;
MPI_Datatype recvtype;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allgatherv()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Allgatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Allreduce( sendbuf, recvbuf, count, datatype, op, comm )
void * sendbuf;
void * recvbuf;
int count;
MPI_Datatype datatype;
MPI_Op op;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allreduce()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Allreduce( sendbuf, recvbuf, count, datatype, op, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Alltoall( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm )
void * sendbuf;
int sendcount;
MPI_Datatype sendtype;
void * recvbuf;
int recvcnt;
MPI_Datatype recvtype;
MPI_Comm comm;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Alltoall()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Alltoall( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Alltoallv( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm )
void * sendbuf;
int * sendcnts;
int * sdispls;
MPI_Datatype sendtype;
void * recvbuf;
int * recvcnts;
int * rdispls;
MPI_Datatype recvtype;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Alltoallv()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Alltoallv( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Barrier( comm )
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Barrier()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Barrier( comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Bcast( buffer, count, datatype, root, comm )
void * buffer;
int count;
MPI_Datatype datatype;
int root;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Bcast()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Bcast( buffer, count, datatype, root, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Gather( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm )
void * sendbuf;
int sendcnt;
MPI_Datatype sendtype;
void * recvbuf;
int recvcount;
MPI_Datatype recvtype;
int root;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Gather()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Gather( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Gatherv( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm )
void * sendbuf;
int sendcnt;
MPI_Datatype sendtype;
void * recvbuf;
int * recvcnts;
int * displs;
MPI_Datatype recvtype;
int root;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Gatherv()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Gatherv( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Op_create( function, commute, op )
MPI_User_function * function;
int commute;
MPI_Op * op;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Op_create()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Op_create( function, commute, op );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Op_free( op )
MPI_Op * op;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Op_free()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Op_free( op );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Reduce_scatter( sendbuf, recvbuf, recvcnts, datatype, op, comm )
void * sendbuf;
void * recvbuf;
int * recvcnts;
MPI_Datatype datatype;
MPI_Op op;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Reduce_scatter()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Reduce_scatter( sendbuf, recvbuf, recvcnts, datatype, op, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Reduce( sendbuf, recvbuf, count, datatype, op, root, comm )
void * sendbuf;
void * recvbuf;
int count;
MPI_Datatype datatype;
MPI_Op op;
int root;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Reduce()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Reduce( sendbuf, recvbuf, count, datatype, op, root, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Scan( sendbuf, recvbuf, count, datatype, op, comm )
void * sendbuf;
void * recvbuf;
int count;
MPI_Datatype datatype;
MPI_Op op;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scan()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Scan( sendbuf, recvbuf, count, datatype, op, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Scatter( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm )
void * sendbuf;
int sendcnt;
MPI_Datatype sendtype;
void * recvbuf;
int recvcnt;
MPI_Datatype recvtype;
int root;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scatter()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Scatter( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Scatterv( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm )
void * sendbuf;
int * sendcnts;
int * displs;
MPI_Datatype sendtype;
void * recvbuf;
int recvcnt;
MPI_Datatype recvtype;
int root;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scatterv()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Scatterv( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Attr_delete( comm, keyval )
MPI_Comm comm;
int keyval;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Attr_delete()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Attr_delete( comm, keyval );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Attr_get( comm, keyval, attr_value, flag )
MPI_Comm comm;
int keyval;
void * attr_value;
int * flag;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Attr_get()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Attr_get( comm, keyval, attr_value, flag );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Attr_put( comm, keyval, attr_value )
MPI_Comm comm;
int keyval;
void * attr_value;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Attr_put()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Attr_put( comm, keyval, attr_value );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_compare( comm1, comm2, result )
MPI_Comm comm1;
MPI_Comm comm2;
int * result;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_compare()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_compare( comm1, comm2, result );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_create( comm, group, comm_out )
MPI_Comm comm;
MPI_Group group;
MPI_Comm * comm_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_create()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_create( comm, group, comm_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_dup( comm, comm_out )
MPI_Comm comm;
MPI_Comm * comm_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_dup()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_dup( comm, comm_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_free( comm )
MPI_Comm * comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_free()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_free( comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_group( comm, group )
MPI_Comm comm;
MPI_Group * group;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_group()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_group( comm, group );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_rank( comm, rank )
MPI_Comm comm;
int * rank;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_rank()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_rank( comm, rank );
  TAU_PROFILE_SET_NODE(*rank);

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_remote_group( comm, group )
MPI_Comm comm;
MPI_Group * group;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_remote_group()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_remote_group( comm, group );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_remote_size( comm, size )
MPI_Comm comm;
int * size;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_remote_size()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_remote_size( comm, size );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_size( comm, size )
MPI_Comm comm;
int * size;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_size()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_size( comm, size );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_split( comm, color, key, comm_out )
MPI_Comm comm;
int color;
int key;
MPI_Comm * comm_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_split()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_split( comm, color, key, comm_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_test_inter( comm, flag )
MPI_Comm comm;
int * flag;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_test_inter()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_test_inter( comm, flag );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_compare( group1, group2, result )
MPI_Group group1;
MPI_Group group2;
int * result;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_compare()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_compare( group1, group2, result );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_difference( group1, group2, group_out )
MPI_Group group1;
MPI_Group group2;
MPI_Group * group_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_difference()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_difference( group1, group2, group_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_excl( group, n, ranks, newgroup )
MPI_Group group;
int n;
int * ranks;
MPI_Group * newgroup;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_excl()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_excl( group, n, ranks, newgroup );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_free( group )
MPI_Group * group;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_free()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_free( group );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_incl( group, n, ranks, group_out )
MPI_Group group;
int n;
int * ranks;
MPI_Group * group_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_incl()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_incl( group, n, ranks, group_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_intersection( group1, group2, group_out )
MPI_Group group1;
MPI_Group group2;
MPI_Group * group_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_intersection()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_intersection( group1, group2, group_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_rank( group, rank )
MPI_Group group;
int * rank;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_rank()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_rank( group, rank );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_range_excl( group, n, ranges, newgroup )
MPI_Group group;
int n;
int ranges[][3];
MPI_Group * newgroup;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_range_excl()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_range_excl( group, n, ranges, newgroup );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_range_incl( group, n, ranges, newgroup )
MPI_Group group;
int n;
int ranges[][3];
MPI_Group * newgroup;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_range_incl()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_range_incl( group, n, ranges, newgroup );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_size( group, size )
MPI_Group group;
int * size;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_size()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_size( group, size );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_translate_ranks( group_a, n, ranks_a, group_b, ranks_b )
MPI_Group group_a;
int n;
int * ranks_a;
MPI_Group group_b;
int * ranks_b;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_translate_ranks()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_translate_ranks( group_a, n, ranks_a, group_b, ranks_b );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_union( group1, group2, group_out )
MPI_Group group1;
MPI_Group group2;
MPI_Group * group_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_union()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_union( group1, group2, group_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Intercomm_create( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out )
MPI_Comm local_comm;
int local_leader;
MPI_Comm peer_comm;
int remote_leader;
int tag;
MPI_Comm * comm_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Intercomm_create()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Intercomm_create( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Intercomm_merge( comm, high, comm_out )
MPI_Comm comm;
int high;
MPI_Comm * comm_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Intercomm_merge()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Intercomm_merge( comm, high, comm_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Keyval_create( copy_fn, delete_fn, keyval, extra_state )
MPI_Copy_function * copy_fn;
MPI_Delete_function * delete_fn;
int * keyval;
void * extra_state;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Keyval_create()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Keyval_create( copy_fn, delete_fn, keyval, extra_state );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Keyval_free( keyval )
int * keyval;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Keyval_free()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Keyval_free( keyval );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Abort( comm, errorcode )
MPI_Comm comm;
int errorcode;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Abort()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Abort( comm, errorcode );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Error_class( errorcode, errorclass )
int errorcode;
int * errorclass;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Error_class()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Error_class( errorcode, errorclass );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Errhandler_create( function, errhandler )
MPI_Handler_function * function;
MPI_Errhandler * errhandler;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_create()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Errhandler_create( function, errhandler );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Errhandler_free( errhandler )
MPI_Errhandler * errhandler;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_free()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Errhandler_free( errhandler );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Errhandler_get( comm, errhandler )
MPI_Comm comm;
MPI_Errhandler * errhandler;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_get()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Errhandler_get( comm, errhandler );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Error_string( errorcode, string, resultlen )
int errorcode;
char * string;
int * resultlen;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Error_string()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Error_string( errorcode, string, resultlen );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Errhandler_set( comm, errhandler )
MPI_Comm comm;
MPI_Errhandler errhandler;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_set()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Errhandler_set( comm, errhandler );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Finalize(  )
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Finalize()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Finalize(  );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Get_processor_name( name, resultlen )
char * name;
int * resultlen;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Get_processor_name()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Get_processor_name( name, resultlen );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Init( argc, argv )
int * argc;
char *** argv;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Init()", " ", TAU_MESSAGE);
  TAU_PROFILE_INIT(*argc, *argv);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Init( argc, argv );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Initialized( flag )
int * flag;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Initialized()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Initialized( flag );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

double  MPI_Wtick(  )
{
  double  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Wtick()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Wtick(  );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

double  MPI_Wtime(  )
{
  double  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Wtime()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Wtime(  );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Address( location, address )
void * location;
MPI_Aint * address;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Address()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Address( location, address );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Bsend( buf, count, datatype, dest, tag, comm )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Bsend()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Bsend( buf, count, datatype, dest, tag, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Bsend_init( buf, count, datatype, dest, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Bsend_init()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Bsend_init( buf, count, datatype, dest, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Buffer_attach( buffer, size )
void * buffer;
int size;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Buffer_attach()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Buffer_attach( buffer, size );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Buffer_detach( buffer, size )
void * buffer;
int * size;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Buffer_detach()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Buffer_detach( buffer, size );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Cancel( request )
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cancel()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cancel( request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Request_free( request )
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Request_free()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Request_free( request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Recv_init( buf, count, datatype, source, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int source;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Recv_init()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Recv_init( buf, count, datatype, source, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Send_init( buf, count, datatype, dest, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Send_init()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Send_init( buf, count, datatype, dest, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Get_elements( status, datatype, elements )
MPI_Status * status;
MPI_Datatype datatype;
int * elements;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Get_elements()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Get_elements( status, datatype, elements );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Get_count( status, datatype, count )
MPI_Status * status;
MPI_Datatype datatype;
int * count;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Get_count()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Get_count( status, datatype, count );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Ibsend( buf, count, datatype, dest, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Ibsend()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Ibsend( buf, count, datatype, dest, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Iprobe( source, tag, comm, flag, status )
int source;
int tag;
MPI_Comm comm;
int * flag;
MPI_Status * status;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Iprobe()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Iprobe( source, tag, comm, flag, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Irecv( buf, count, datatype, source, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int source;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Irecv()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Irecv( buf, count, datatype, source, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Irsend( buf, count, datatype, dest, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Irsend()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Irsend( buf, count, datatype, dest, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Isend( buf, count, datatype, dest, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Isend()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Isend( buf, count, datatype, dest, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Issend( buf, count, datatype, dest, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Issend()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Issend( buf, count, datatype, dest, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Pack( inbuf, incount, type, outbuf, outcount, position, comm )
void * inbuf;
int incount;
MPI_Datatype type;
void * outbuf;
int outcount;
int * position;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Pack()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Pack( inbuf, incount, type, outbuf, outcount, position, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Pack_size( incount, datatype, comm, size )
int incount;
MPI_Datatype datatype;
MPI_Comm comm;
int * size;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Pack_size()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Pack_size( incount, datatype, comm, size );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Probe( source, tag, comm, status )
int source;
int tag;
MPI_Comm comm;
MPI_Status * status;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Probe()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Probe( source, tag, comm, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}


int  MPI_Rsend( buf, count, datatype, dest, tag, comm )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Rsend()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Rsend( buf, count, datatype, dest, tag, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Rsend_init( buf, count, datatype, dest, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Rsend_init()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Rsend_init( buf, count, datatype, dest, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}


int  MPI_Sendrecv( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status )
void * sendbuf;
int sendcount;
MPI_Datatype sendtype;
int dest;
int sendtag;
void * recvbuf;
int recvcount;
MPI_Datatype recvtype;
int source;
int recvtag;
MPI_Comm comm;
MPI_Status * status;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Sendrecv()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Sendrecv( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Sendrecv_replace( buf, count, datatype, dest, sendtag, source, recvtag, comm, status )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int sendtag;
int source;
int recvtag;
MPI_Comm comm;
MPI_Status * status;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Sendrecv_replace()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Sendrecv_replace( buf, count, datatype, dest, sendtag, source, recvtag, comm, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Ssend( buf, count, datatype, dest, tag, comm )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Ssend()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Ssend( buf, count, datatype, dest, tag, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Ssend_init( buf, count, datatype, dest, tag, comm, request )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Ssend_init()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Ssend_init( buf, count, datatype, dest, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Start( request )
MPI_Request * request;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Start()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Start( request );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Startall( count, array_of_requests )
int count;
MPI_Request * array_of_requests;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Startall()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Startall( count, array_of_requests );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Test( request, flag, status )
MPI_Request * request;
int * flag;
MPI_Status * status;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Test()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Test( request, flag, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Testall( count, array_of_requests, flag, array_of_statuses )
int count;
MPI_Request * array_of_requests;
int * flag;
MPI_Status * array_of_statuses;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Testall()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Testall( count, array_of_requests, flag, array_of_statuses );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Testany( count, array_of_requests, index, flag, status )
int count;
MPI_Request * array_of_requests;
int * index;
int * flag;
MPI_Status * status;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Testany()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Testany( count, array_of_requests, index, flag, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Test_cancelled( status, flag )
MPI_Status * status;
int * flag;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Test_cancelled()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Test_cancelled( status, flag );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Testsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses )
int incount;
MPI_Request * array_of_requests;
int * outcount;
int * array_of_indices;
MPI_Status * array_of_statuses;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Testsome()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Testsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Type_commit( datatype )
MPI_Datatype * datatype;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_commit()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_commit( datatype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Type_contiguous( count, old_type, newtype )
int count;
MPI_Datatype old_type;
MPI_Datatype * newtype;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_contiguous()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_contiguous( count, old_type, newtype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

/* Not on SGI
int   MPI_Type_count( datatype, count )
MPI_Datatype datatype;
int * count;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_count()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_count( datatype, count );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
*/

int  MPI_Type_extent( datatype, extent )
MPI_Datatype datatype;
MPI_Aint * extent;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_extent()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_extent( datatype, extent );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Type_free( datatype )
MPI_Datatype * datatype;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_free()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_free( datatype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Type_hindexed( count, blocklens, indices, old_type, newtype )
int count;
int * blocklens;
MPI_Aint * indices;
MPI_Datatype old_type;
MPI_Datatype * newtype;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_hindexed()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_hindexed( count, blocklens, indices, old_type, newtype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Type_hvector( count, blocklen, stride, old_type, newtype )
int count;
int blocklen;
MPI_Aint stride;
MPI_Datatype old_type;
MPI_Datatype * newtype;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_hvector()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_hvector( count, blocklen, stride, old_type, newtype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Type_indexed( count, blocklens, indices, old_type, newtype )
int count;
int * blocklens;
int * indices;
MPI_Datatype old_type;
MPI_Datatype * newtype;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_indexed()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_indexed( count, blocklens, indices, old_type, newtype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Type_lb( datatype, displacement )
MPI_Datatype datatype;
MPI_Aint * displacement;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_lb()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_lb( datatype, displacement );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Type_size( datatype, size )
MPI_Datatype datatype;
int * size;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_size()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_size( datatype, size );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Type_struct( count, blocklens, indices, old_types, newtype )
int count;
int * blocklens;
MPI_Aint * indices;
MPI_Datatype * old_types;
MPI_Datatype * newtype;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_struct()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_struct( count, blocklens, indices, old_types, newtype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Type_ub( datatype, displacement )
MPI_Datatype datatype;
MPI_Aint * displacement;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_ub()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_ub( datatype, displacement );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Type_vector( count, blocklen, stride, old_type, newtype )
int count;
int blocklen;
int stride;
MPI_Datatype old_type;
MPI_Datatype * newtype;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_vector()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_vector( count, blocklen, stride, old_type, newtype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Unpack( inbuf, insize, position, outbuf, outcount, type, comm )
void * inbuf;
int insize;
int * position;
void * outbuf;
int outcount;
MPI_Datatype type;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Unpack()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Unpack( inbuf, insize, position, outbuf, outcount, type, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Wait( request, status )
MPI_Request * request;
MPI_Status * status;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Wait()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Wait( request, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Waitall( count, array_of_requests, array_of_statuses )
int count;
MPI_Request * array_of_requests;
MPI_Status * array_of_statuses;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Waitall()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Waitall( count, array_of_requests, array_of_statuses );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Waitany( count, array_of_requests, index, status )
int count;
MPI_Request * array_of_requests;
int * index;
MPI_Status * status;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Waitany()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Waitany( count, array_of_requests, index, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Waitsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses )
int incount;
MPI_Request * array_of_requests;
int * outcount;
int * array_of_indices;
MPI_Status * array_of_statuses;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Waitsome()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Waitsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_coords( comm, rank, maxdims, coords )
MPI_Comm comm;
int rank;
int maxdims;
int * coords;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_coords()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cart_coords( comm, rank, maxdims, coords );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_create( comm_old, ndims, dims, periods, reorder, comm_cart )
MPI_Comm comm_old;
int ndims;
int * dims;
int * periods;
int reorder;
MPI_Comm * comm_cart;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_create()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cart_create( comm_old, ndims, dims, periods, reorder, comm_cart );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_get( comm, maxdims, dims, periods, coords )
MPI_Comm comm;
int maxdims;
int * dims;
int * periods;
int * coords;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_get()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cart_get( comm, maxdims, dims, periods, coords );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_map( comm_old, ndims, dims, periods, newrank )
MPI_Comm comm_old;
int ndims;
int * dims;
int * periods;
int * newrank;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_map()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cart_map( comm_old, ndims, dims, periods, newrank );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_rank( comm, coords, rank )
MPI_Comm comm;
int * coords;
int * rank;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_rank()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cart_rank( comm, coords, rank );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_shift( comm, direction, displ, source, dest )
MPI_Comm comm;
int direction;
int displ;
int * source;
int * dest;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_shift()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cart_shift( comm, direction, displ, source, dest );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_sub( comm, remain_dims, comm_new )
MPI_Comm comm;
int * remain_dims;
MPI_Comm * comm_new;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_sub()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cart_sub( comm, remain_dims, comm_new );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cartdim_get( comm, ndims )
MPI_Comm comm;
int * ndims;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cartdim_get()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Cartdim_get( comm, ndims );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Dims_create( nnodes, ndims, dims )
int nnodes;
int ndims;
int * dims;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Dims_create()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Dims_create( nnodes, ndims, dims );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Graph_create( comm_old, nnodes, index, edges, reorder, comm_graph )
MPI_Comm comm_old;
int nnodes;
int * index;
int * edges;
int reorder;
MPI_Comm * comm_graph;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_create()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Graph_create( comm_old, nnodes, index, edges, reorder, comm_graph );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Graph_get( comm, maxindex, maxedges, index, edges )
MPI_Comm comm;
int maxindex;
int maxedges;
int * index;
int * edges;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_get()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Graph_get( comm, maxindex, maxedges, index, edges );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Graph_map( comm_old, nnodes, index, edges, newrank )
MPI_Comm comm_old;
int nnodes;
int * index;
int * edges;
int * newrank;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_map()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Graph_map( comm_old, nnodes, index, edges, newrank );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Graph_neighbors( comm, rank, maxneighbors, neighbors )
MPI_Comm comm;
int rank;
int  maxneighbors;
int * neighbors;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_neighbors()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Graph_neighbors( comm, rank, maxneighbors, neighbors );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Graph_neighbors_count( comm, rank, nneighbors )
MPI_Comm comm;
int rank;
int * nneighbors;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_neighbors_count()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Graph_neighbors_count( comm, rank, nneighbors );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Graphdims_get( comm, nnodes, nedges )
MPI_Comm comm;
int * nnodes;
int * nedges;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Graphdims_get()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Graphdims_get( comm, nnodes, nedges );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Topo_test( comm, top_type )
MPI_Comm comm;
int * top_type;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Topo_test()", " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Topo_test( comm, top_type );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
/* End of wrapper */
