#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#define TAU_MAX_REQUESTS  4096

extern void TauSyncClocks(int rank, int size);
extern void TauSyncFinalClocks(int rank, int size);
extern int Tau_write_snapshot(const char *name, int finalize);
extern int Tau_mergeProfiles();


/* This file uses the MPI Profiling Interface with TAU instrumentation.
   It has been adopted from the MPE Profiling interface wrapper generator
   wrappergen that is part of the MPICH distribution. It differs from MPE
   in where the calls are placed. For e.g., in TAU a send is traced before
   the MPI_Send and a receive after MPI_Recv. This avoids -ve time problems
   that can happen on a uniprocessor if a receive is traced before the send
   is traced. 

   This file was once generated using:
   % <mpich>/mpe/profiling/wrappergen/wrappergen -w TauMpi.w -o TauMpi.c

*/

/* Requests */

#ifndef TAU_NOCOMM
#define TAU_TRACK_MSG 1
#endif /* TAU_NOCOMM */


#ifdef TAU_TRACK_MSG
typedef struct request_list_ {
    MPI_Request request; /* SSS request should be a pointer */
    int         status, size, tag, otherParty;
    int         is_persistent;
    MPI_Comm    comm;
    struct request_list_ *next;
} request_list;

#define RQ_SEND    0x1
#define RQ_RECV    0x2
#define RQ_CANCEL  0x4
/* if MPI_Cancel is called on a request, 'or' RQ_CANCEL into status.
** After a Wait* or Test* is called on that request, check for RQ_CANCEL.
** If the bit is set, check with MPI_Test_cancelled before registering
** the send/receive as 'happening'.
**
*/

#define rq_alloc( head_alloc, newrq ) {\
      if (head_alloc) {\
        newrq=head_alloc;head_alloc=newrq->next;\
	}else{\
      newrq = (request_list*) malloc(sizeof( request_list ));\
      }}

#define rq_remove_at( head, tail, head_alloc, ptr, last ) { \
  if (ptr) { \
    if (!last) { \
      head = ptr->next; \
    } else { \
      last->next = ptr->next; \
      if (tail == ptr) tail = last; \
    } \
	  ptr->next = head_alloc; head_alloc = ptr;}}

#define rq_remove( head, tail, head_alloc, rq ) { \
  request_list *ptr, *last; \
  ptr = head; \
  last = 0; \
  while (ptr && (ptr->request != rq)) { \
    last = ptr; \
    ptr = ptr->next; \
  } \
	rq_remove_at( head, tail, head_alloc, ptr, last );}


#define rq_add( head, tail, rq ) { \
  if (!head) { \
    head = tail = rq; \
  } else { \
    tail->next = rq; tail = rq; \
  }}

#define rq_find( head, req, rq ) { \
  rq = head; \
  while (rq && (rq->request != req)) rq = rq->next; }

#define rq_init( head_alloc ) {\
  int i; request_list *newrq; head_alloc = 0;\
  for (i=0;i<20;i++) {\
      newrq = (request_list*) malloc(sizeof( request_list ));\
      newrq->next = head_alloc;\
      head_alloc = newrq;\
  }}

#define rq_end( head_alloc ) {\
  request_list *rq; while (head_alloc) {\
	rq = head_alloc->next;free(head_alloc);head_alloc=rq;}}
static request_list *requests_head_0=NULL, *requests_tail_0=NULL;
#endif /* TAU_TRACK_MSG */
static int procid_0;

/* This function translates a given rank in a given communicator to the proper
   rank in MPI_COMM_WORLD */
static int translateRankToWorld(MPI_Comm comm, int rank) {
  MPI_Group commGroup, worldGroup;
  int ranks[1], worldranks[1];
  if (comm != MPI_COMM_WORLD) {

    int result;
    PMPI_Comm_compare(comm, MPI_COMM_WORLD, &result);
    if (result == MPI_IDENT || result == MPI_CONGRUENT) {
      return rank;
    } else {
      ranks[0] = rank;
      PMPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
      PMPI_Comm_group(comm, &commGroup);
      PMPI_Group_translate_ranks(commGroup, 1, ranks, worldGroup, worldranks);
      return worldranks[0];
    }
  }
  return rank;
}

/* MPI PROFILING INTERFACE WRAPPERS BEGIN HERE */

/* Message_prof keeps track of when sends and receives 'happen'.  The
** time that each send or receive happens is different for each type of
** send or receive.
**
** Check for MPI_PROC_NULL
**
**   definitely a send:
**     Before a call to MPI_Send.
**     Before a call to MPI_Bsend.
**     Before a call to MPI_Ssend.
**     Before a call to MPI_Rsend.
**
**
**   definitely a receive:
**     After a call to MPI_Recv.
**
**   definitely a send before and a receive after :
**     a call to MPI_Sendrecv
**     a call to MPI_Sendrecv_replace
**
**   maybe a send, maybe a receive:
**     Before a call to MPI_Wait.
**     Before a call to MPI_Waitany.
**     Before a call to MPI_Waitsome.
**     Before a call to MPI_Waitall.
**     After a call to MPI_Probe
**   maybe neither:
**     Before a call to MPI_Test.
**     Before a call to MPI_Testany.
**     Before a call to MPI_Testsome.
**     Before a call to MPI_Testall.
**     After a call to MPI_Iprobe
**
**   start request for a send:
**     After a call to MPI_Isend.
**     After a call to MPI_Ibsend.
**     After a call to MPI_Issend.
**     After a call to MPI_Irsend.
**     After a call to MPI_Send_init.
**     After a call to MPI_Bsend_init.
**     After a call to MPI_Ssend_init.
**     After a call to MPI_Rsend_init.
**
**   start request for a recv:
**     After a call to MPI_Irecv.
**     After a call to MPI_Recv_init.
**
**   stop watching a request:
**     Before a call to MPI_Request_free
**
**   mark a request as possible cancelled:
**     After a call to MPI_Cancel
**
*/


  

int TAUDECL tau_totalnodes(int set_or_get, int value);


#ifdef TAU_TRACK_MSG

void TauProcessRecv ( request, status, note )
MPI_Request request;
MPI_Status *status;
char *note;
{
  request_list *rq, *last;
  int otherid, othertag;

#ifdef DEBUG
  int myrank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif /* DEBUG */

  /* look for request */
  rq = requests_head_0;
  last = 0;

  /* first request */
  while ((rq != NULL) && (rq->request != request)) {
#ifdef DEBUG 
   printf("Node %d: Comparing %lx %lx\n", myrank, rq->request, request);
#endif /* DEBUG */

    last = rq;
    rq = rq->next;
  }

  if (!rq) {
#ifdef DEBUG
    fprintf( stderr, "Node %d: Request not found in '%s'.\n",myrank, note );
#endif /* DEBUG */
    return ;                /* request not found */
  }
#ifdef DEBUG
  else
  {
    printf("Node %d: Request found %lx\n", myrank, request);
  }
#endif /* DEBUG */
  /* We post a receive here */
  if ((rq) && rq->status == RQ_RECV)
  { /* See if we need to see the status to get values of tag & id */
    /* for wildcard receives from any task */
    /* if (rq->otherParty == MPI_ANY_SOURCE) */ 
    otherid = status->MPI_SOURCE;
    /* if (rq->tag == MPI_ANY_TAG) */
    othertag = status->MPI_TAG;
    /* post the receive message */
    TAU_TRACE_RECVMSG(othertag, translateRankToWorld(rq->comm, otherid), rq->size);
  }

  if (rq->is_persistent == 0) {
    /* Remove the record from the request list */
    if (last) {
      if (rq == requests_tail_0) {
	requests_tail_0 = last;
      }
      last->next = rq->next;
    } else {
      requests_head_0 = rq->next;
    }
    free( rq );
  }
  
  return ; 
}

/* This routine traverses the list of requests and checks for RQ_SEND. The 
   message is logged if this request matches */


void TauProcessSend ( request, note )
MPI_Request request;
char *note;
{
  request_list *rq, *last;
  int otherid, othertag;

#ifdef DEBUG
  int myrank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif /* DEBUG */

  /* look for request */
  rq = requests_head_0;
  last = 0;

  /* first request */
  while ((rq != NULL) && (rq->request != request)) {
#ifdef DEBUG 
   printf("Node %d: Comparing %lx %lx\n", myrank, rq->request, request);
#endif /* DEBUG */

    last = rq;
    rq = rq->next;
  }

  if (!rq) {
#ifdef DEBUG
    fprintf( stderr, "Node %d: Request not found in '%s'.\n",myrank, note );
#endif /* DEBUG */
    return ;                /* request not found */
  }
#ifdef DEBUG
  else
  {
    printf("Node %d: Request found %lx\n", myrank, request);
  }
#endif /* DEBUG */
  if ((rq) && rq->status == RQ_SEND)
  { 
    otherid = translateRankToWorld(rq->comm, rq->otherParty);
    othertag = rq->tag;
    /* post the send message */
    TAU_TRACE_SENDMSG(othertag, otherid, rq->size);
  }

  return ; 
}




request_list *TauGetRequest( MPI_Request request) {
  request_list *rq;

  rq = requests_head_0;

  while ((rq != NULL) && (rq->request != request)) {
    rq = rq->next;
  }
  return rq;
}


void TauAddRequest (int status, int count, MPI_Datatype datatype, int other, 
		    int tag, MPI_Comm comm, MPI_Request *request, int returnVal, int persistent) {
  int typesize;
  request_list *newrq1;
  if (other != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    if (newrq1 = (request_list*) malloc(sizeof( request_list ))) {
      PMPI_Type_size( datatype, &typesize );
      newrq1->request = *request;
      newrq1->status = status;
      newrq1->size = typesize * count;
      newrq1->otherParty = other;
      newrq1->comm = comm;
      newrq1->tag = tag;
      newrq1->is_persistent = persistent;
      newrq1->next = 0;
      rq_add( requests_head_0, requests_tail_0, newrq1 );
    }
  }
}

/* This routine traverses the list of requests and deletes the given request */
void TauRemoveRequest ( request, note )
MPI_Request request;
char *note;
{
  request_list *rq, *last;

#ifdef DEBUG
  int myrank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif /* DEBUG */

  /* look for request */
  rq = requests_head_0;
  last = 0;

  /* first request */
  while ((rq != NULL) && (rq->request != request)) {
#ifdef DEBUG
   printf("Node %d: Comparing %lx %lx\n", myrank, rq->request, request);
#endif /* DEBUG */ 
    last = rq;
    rq = rq->next;
  }

  if (!rq) {
#ifdef DEBUG
    fprintf( stderr, "Node %d: Request not found in '%s'.\n",myrank, note );
#endif /* DEBUG */
    return ;		/* request not found */
  }
  /* remove the request */
  if (last) {
    if (rq == requests_tail_0) {
      requests_tail_0 = last;
    }
    last->next = rq->next;
  } else {
    requests_head_0 = rq->next;
  }
  free( rq );
   
  return ; 
}

#endif /* TAU_TRACK_MSG */





/* NOTE: MPI_Type_count was not implemented in mpich-1.2.0. Remove it from this
   list when it is implemented in libpmpich.a */









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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allgather()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Allgather( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm );
  PMPI_Type_size( recvtype, &typesize );
  TAU_ALLGATHER_DATA(typesize*recvcount);

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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allgatherv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Allgatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm );
  PMPI_Type_size( recvtype, &typesize );
  TAU_ALLGATHER_DATA(typesize*(*recvcounts));
  

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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allreduce()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Allreduce( sendbuf, recvbuf, count, datatype, op, comm );
  PMPI_Type_size( datatype, &typesize );
  TAU_ALLREDUCE_DATA(typesize*count);

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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Alltoall()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Alltoall( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm );
  PMPI_Type_size( sendtype, &typesize );
  TAU_ALLTOALL_DATA(typesize*sendcount);

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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Alltoallv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Alltoallv( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm );
  PMPI_Type_size( sendtype, &typesize );
  TAU_ALLTOALL_DATA(typesize*(*sendcnts));

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Barrier( comm )
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Barrier()",  " ", TAU_MESSAGE);
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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Bcast()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Bcast( buffer, count, datatype, root, comm );
  PMPI_Type_size( datatype, &typesize );
  TAU_BCAST_DATA(typesize*count);

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
  int   typesize;
  int   rank;

  TAU_PROFILE_TIMER(tautimer, "MPI_Gather()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Gather( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm );

  PMPI_Comm_rank ( comm, &rank );
  if (rank == root) {
    PMPI_Type_size( recvtype, &typesize );
    TAU_GATHER_DATA(typesize*recvcount);
  }


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
  int   typesize;
  int   rank;

  TAU_PROFILE_TIMER(tautimer, "MPI_Gatherv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Gatherv( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm );

  PMPI_Comm_rank ( comm, &rank );
  if (rank == root) {
    PMPI_Type_size( recvtype, &typesize );
    TAU_GATHER_DATA(typesize*(*recvcnts));
  }

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Op_create( function, commute, op )
MPI_User_function * function;
int commute;
MPI_Op * op;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Op_create()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Op_create( function, commute, op );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Op_free( op )
MPI_Op * op;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Op_free()",  " ", TAU_MESSAGE);
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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Reduce_scatter()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Reduce_scatter( sendbuf, recvbuf, recvcnts, datatype, op, comm );
  PMPI_Type_size( datatype, &typesize );
  TAU_REDUCESCATTER_DATA(typesize*(*recvcnts));

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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Reduce()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Reduce( sendbuf, recvbuf, count, datatype, op, root, comm );
  PMPI_Type_size( datatype, &typesize );
  TAU_REDUCE_DATA(typesize*count);

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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scan()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Scan( sendbuf, recvbuf, count, datatype, op, comm );
  PMPI_Type_size( datatype, &typesize );
  TAU_SCAN_DATA(typesize*count);

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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scatter()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Scatter( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm );
  PMPI_Type_size( sendtype, &typesize );
  TAU_SCATTER_DATA(typesize*sendcnt);

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
  int   typesize;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scatterv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Scatterv( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm );
  PMPI_Type_size( sendtype, &typesize );
  TAU_SCATTER_DATA(typesize*(*sendcnts));

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Attr_delete( comm, keyval )
MPI_Comm comm;
int keyval;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Attr_delete()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Attr_get()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Attr_put()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_compare()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_create()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_dup()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_dup( comm, comm_out );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_free( comm )
MPI_Comm * comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_free()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_group()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_rank()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Comm_rank( comm, rank );

  TAU_PROFILE_STOP(tautimer);

  /* Set the node as we did in MPI_Init */
  if (comm == MPI_COMM_WORLD)
    TAU_PROFILE_SET_NODE(*rank);

  return returnVal;
}

int   MPI_Comm_remote_group( comm, group )
MPI_Comm comm;
MPI_Group * group;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_remote_group()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_remote_size()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_size()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_split()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_test_inter()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_compare()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_difference()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_excl()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Group_excl( group, n, ranks, newgroup );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_free( group )
MPI_Group * group;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_free()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_incl()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_intersection()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_rank()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_range_excl()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_range_incl()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_size()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_translate_ranks()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_union()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Intercomm_create()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Intercomm_merge()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Keyval_create()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Keyval_create( copy_fn, delete_fn, keyval, extra_state );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Keyval_free( keyval )
int * keyval;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Keyval_free()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Keyval_free( keyval );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

/* LAM MPI defines MPI_Abort as a macro! We check for this and if it is
   defined that way, we change the MPI_Abort wrapper */
#if (defined(MPI_Abort) && defined(_ULM_MPI_H_))
int _MPI_Abort( MPI_Comm comm, int errorcode, char * file, int line)
#else
int  MPI_Abort( comm, errorcode )
MPI_Comm comm;
int errorcode;
#endif /* MPI_Abort & LAM MPI [LAM MPI] */
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Abort()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Error_class()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_create()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Errhandler_create( function, errhandler );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Errhandler_free( errhandler )
MPI_Errhandler * errhandler;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_free()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_get()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Error_string()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_set()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Errhandler_set( comm, errhandler );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Finalize(  )
{
  int  returnVal;
  int size;
  char procname[MPI_MAX_PROCESSOR_NAME];
  int  procnamelength;

  TAU_PROFILE_TIMER(tautimer, "MPI_Finalize()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  

  if (TauEnv_get_synchronize_clocks()) {
    PMPI_Comm_size( MPI_COMM_WORLD, &size );
    TauSyncFinalClocks(procid_0, size);
  }

  PMPI_Get_processor_name(procname, &procnamelength);
  TAU_METADATA("MPI Processor Name", procname);

  /* Grab the node id, we don't always wrap mpi_init */
  PMPI_Comm_rank( MPI_COMM_WORLD, &procid_0 );
  TAU_PROFILE_SET_NODE(procid_0 ); 

  /* Create a merged profile if requested */
  if (TauEnv_get_profile_format() == TAU_FORMAT_MERGED) {
    Tau_write_snapshot("merge", 1);
    Tau_mergeProfiles();
  }

  returnVal = PMPI_Finalize();

  TAU_PROFILE_STOP(tautimer);

  Tau_stop_top_level_timer_if_necessary();
  return returnVal;
}

int  MPI_Get_processor_name( name, resultlen )
char * name;
int * resultlen;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Get_processor_name()",  " ", TAU_MESSAGE);
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
  int  size;
  char procname[MPI_MAX_PROCESSOR_NAME];
  int  procnamelength;

  TAU_PROFILE_TIMER(tautimer, "MPI_Init()",  " ", TAU_MESSAGE); 
  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Init( argc, argv );

  TAU_PROFILE_STOP(tautimer); 

  PMPI_Comm_rank( MPI_COMM_WORLD, &procid_0 );
  TAU_PROFILE_SET_NODE(procid_0 ); 

  PMPI_Comm_size( MPI_COMM_WORLD, &size );
  tau_totalnodes(1, size); /* Set the totalnodes */

  PMPI_Get_processor_name(procname, &procnamelength);
  TAU_METADATA("MPI Processor Name", procname);

  if (TauEnv_get_synchronize_clocks()) {
    TauSyncClocks(procid_0, size);
  }

#ifdef TAU_TRACK_MSG
  requests_head_0 = requests_tail_0 = 0;
#endif /* TAU_TRACK_MSG */

  return returnVal;
}

#ifdef TAU_MPI_THREADED

int  MPI_Init_thread (argc, argv, required, provided )
int * argc;
char *** argv;
int required;
int *provided;
{
  int  returnVal;
  int  size;
  char procname[MPI_MAX_PROCESSOR_NAME];
  int  procnamelength;

 
  TAU_PROFILE_TIMER(tautimer, "MPI_Init_thread()",  " ", TAU_MESSAGE);
  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_START(tautimer);
 
  returnVal = PMPI_Init_thread( argc, argv, required, provided );

  TAU_PROFILE_STOP(tautimer);

  PMPI_Comm_rank( MPI_COMM_WORLD, &procid_0 );
  TAU_PROFILE_SET_NODE(procid_0 );

  PMPI_Comm_size( MPI_COMM_WORLD, &size );
  tau_totalnodes(1, size); /* Set the totalnodes */

  PMPI_Get_processor_name(procname, &procnamelength);
  TAU_METADATA("MPI Processor Name", procname);

  if (TauEnv_get_synchronize_clocks()) {
    TauSyncClocks(procid_0, size);
  }

#ifdef TAU_TRACK_MSG
  requests_head_0 = requests_tail_0 = 0;
#endif /* TAU_TRACK_MSG */

  return returnVal;
}
#endif /* TAU_MPI_THREADED */



/*
int  MPI_Initialized( flag )
int * flag;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Initialized()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Initialized( flag );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
*/

#ifdef TAU_ENABLE_MPI_WTIME
double  MPI_Wtick(  )
{
  double  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Wtick()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Wtick(  );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

double  MPI_Wtime(  )
{
  double  returnVal;

  /* To enable the instrumentation change group to TAU_MESSAGE */
  TAU_PROFILE_TIMER(tautimer, "MPI_Wtime()",  " ", TAU_DISABLE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Wtime(  );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
#endif

int  MPI_Address( location, address )
void * location;
MPI_Aint * address;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Address()",  " ", TAU_MESSAGE);
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
  int typesize;

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Bsend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize );
    TAU_TRACE_SENDMSG(tag, translateRankToWorld(comm, dest), typesize*count);
    /*
    prof_send( procid_0, dest, tag, typesize*count,
	       "MPI_Bsend" );
    */
  }
#endif /* TAU_TRACK_MSG */
  
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

/* fprintf( stderr, "MPI_Bsend_init call on %d\n", procid_0 ); */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Bsend_init()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Bsend_init( buf, count, datatype, dest, tag, comm, request );

#ifdef TAU_TRACK_MSG
  TauAddRequest(RQ_SEND, count, datatype, dest, tag, comm, request, returnVal, 1);
#endif /* TAU_TRACK_MSG */

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Buffer_attach( buffer, size )
void * buffer;
int size;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Buffer_attach()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Buffer_detach()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Buffer_detach( buffer, size );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Cancel( request )
MPI_Request * request;
{
  int  returnVal;
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Cancel()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
#ifdef TAU_TRACK_MSG
  TauRemoveRequest(*request, "MPI_Cancel");
#endif /* TAU_TRACK_MSG */

  returnVal = PMPI_Cancel( request );

  TAU_PROFILE_STOP(tautimer);


  return returnVal;
}

int  MPI_Request_free( request )
MPI_Request * request;
{
  int  returnVal;


  /* The request may have completed, may have not.  */
  /* We'll assume it didn't. */
  
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Request_free()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_TRACK_MSG
  TauRemoveRequest(*request, "MPI_Request_free");
#endif /* TAU_TRACK_MSG */
  
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
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Recv_init()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Recv_init( buf, count, datatype, source, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

#ifdef TAU_TRACK_MSG
  TauAddRequest(RQ_RECV, count, datatype, source, tag, comm, request, returnVal, 1);
#endif /* TAU_TRACK_MSG */

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

#ifdef DEBUG
  fprintf( stderr, "MPI_Send_init call on %d\n", procid_0 ); 
#endif /* DEBUG */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Send_init()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Send_init( buf, count, datatype, dest, tag, comm, request );

  /* we need to store the request and associate it with the size/tag so MPI_Start can 
     retrieve it and log the TAU_TRACE_SENDMSG */
#ifdef TAU_TRACK_MSG
  TauAddRequest(RQ_SEND, count, datatype, dest, tag, comm, request, returnVal, 1);
#endif /* TAU_TRACK_MSG */

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Get_elements( status, datatype, elements )
MPI_Status * status;
MPI_Datatype datatype;
int * elements;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Get_elements()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Get_count()",  " ", TAU_MESSAGE);
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
#ifdef TAU_TRACK_MSG
  int typesize;
#endif /* TAU_TRACK_MSG */

  
  
/* fprintf( stderr, "MPI_Ibsend call on %d\n", procid_0 ); */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Ibsend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize );
    TAU_TRACE_SENDMSG(tag, translateRankToWorld(comm, dest), count * typesize);
  }
#endif /* TAU_TRACK_MSG */

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

  TAU_PROFILE_TIMER(tautimer, "MPI_Iprobe()",  " ", TAU_MESSAGE);
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
#ifdef TAU_TRACK_MSG
  request_list *newrq1;
  int typesize;

#ifdef DEBUG
  int myrank;
#endif /* DEBUG */

#endif /* TAU_TRACK_MSG */

  
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Irecv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
#ifdef DEBUG
  PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif /* DEBUG */

  returnVal = PMPI_Irecv( buf, count, datatype, source, tag, comm, request );

#ifdef DEBUG
  printf("Node: %d: Irecv: request = %lx\n", myrank, *request);
#endif /* DEBUG */

  TAU_PROFILE_STOP(tautimer);

#ifdef TAU_TRACK_MSG
  if (source != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    if (newrq1 = (request_list*) malloc(sizeof( request_list ))) {
      PMPI_Type_size( datatype, &typesize );
      newrq1->request = *request;
      newrq1->status = RQ_RECV;
      newrq1->size = typesize * count;
      newrq1->otherParty = source;
      newrq1->comm = comm;
      newrq1->tag = tag;
      newrq1->is_persistent = 0;
      newrq1->next = 0;
      rq_add( requests_head_0, requests_tail_0, newrq1 );
    }
  }
#endif /* TAU_TRACK_MSG */

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
#ifdef TAU_TRACK_MSG
  int typesize3;
#endif /* TAU_TRACK_MSG */

  
  
/* fprintf( stderr, "MPI_Irsend call on %d\n", procid_0 ); */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Irsend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize3 );
    TAU_TRACE_SENDMSG(tag, translateRankToWorld(comm, dest), count * typesize3);
  }
#endif /* TAU_TRACK_MSG */

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
#ifdef TAU_TRACK_MSG
  int typesize3;
#endif /* TAU_TRACK_MSG */

  
  
/* fprintf( stderr, "MPI_Isend call on %d\n", procid_0 ); */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Isend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize3 );
    TAU_TRACE_SENDMSG(tag, translateRankToWorld(comm, dest), count * typesize3);
  }
#endif /* TAU_TRACK_MSG */
  
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
#ifdef TAU_TRACK_MSG
  int typesize3;
#endif /* TAU_TRACK_MSG */

  
  
/* fprintf( stderr, "MPI_Issend call on %d\n", procid_0 ); */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Issend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize3 );
    TAU_TRACE_SENDMSG(tag, translateRankToWorld(comm, dest), count * typesize3);
  }
#endif /* TAU_TRACK_MSG */

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

  TAU_PROFILE_TIMER(tautimer, "MPI_Pack()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Pack_size()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Probe()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Probe( source, tag, comm, status );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Recv( buf, count, datatype, source, tag, comm, status )
void * buf;
int count;
MPI_Datatype datatype;
int source;
int tag;
MPI_Comm comm;
MPI_Status * status;
{
  MPI_Status local_status;
  int  returnVal;
  int size;

  TAU_PROFILE_TIMER(tautimer, "MPI_Recv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_TRACK_MSG
  if (status == MPI_STATUS_IGNORE) {
    status = &local_status;
  }
#endif /* TAU_TRACK_MSG */
  
  returnVal = PMPI_Recv( buf, count, datatype, source, tag, comm, status );

#ifdef TAU_TRACK_MSG
  if (source != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    PMPI_Get_count( status, MPI_BYTE, &size );

    /* note that status->MPI_COMM must == comm */
    TAU_TRACE_RECVMSG(status->MPI_TAG,  translateRankToWorld(comm, status->MPI_SOURCE), size);
    /*
    prof_recv( procid_0, status->MPI_SOURCE,
	       status->MPI_TAG, size, "MPI_Recv" );
    */
  }
#endif /* TAU_TRACK_MSG */
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
#ifdef TAU_TRACK_MSG
  int typesize;
#endif /* TAU_TRACK_MSG */

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Rsend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize );
    TAU_TRACE_SENDMSG(tag, translateRankToWorld(comm, dest), typesize*count);
    /*
    prof_send( procid_0, dest, tag, typesize*count,
	       "MPI_Rsend" );
    */
  }
#endif /* TAU_TRACK_MSG */
  
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
  
/* fprintf( stderr, "MPI_Rsend_init call on %d\n", procid_0 ); */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Rsend_init()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Rsend_init( buf, count, datatype, dest, tag, comm, request );

#ifdef TAU_TRACK_MSG
  TauAddRequest(RQ_SEND, count, datatype, dest, tag, comm, request, returnVal, 1);
#endif /* TAU_TRACK_MSG */

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}


int  MPI_Send( buf, count, datatype, dest, tag, comm )
void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
{
  int  returnVal;
#ifdef TAU_TRACK_MSG
  int typesize;
#endif /* TAU_TRACK_MSG */

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Send()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize );

    TAU_TRACE_SENDMSG(tag, translateRankToWorld(comm, dest), typesize*count);
    /*
    prof_send( procid_0, dest, tag, typesize*count,
	       "MPI_Send" );
    */
  }
#endif /* TAU_TRACK_MSG */
  
  returnVal = PMPI_Send( buf, count, datatype, dest, tag, comm );

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
#ifdef TAU_TRACK_MSG
  MPI_Status local_status;
  int typesize1;
  int count;
#endif /* TAU_TRACK_MSG */

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Sendrecv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( sendtype, &typesize1 );
    TAU_TRACE_SENDMSG(sendtag, translateRankToWorld(comm, dest), typesize1*sendcount);
    /*
    prof_send( procid_0, dest, sendtag,
               typesize1*sendcount, "MPI_Sendrecv" );
    */
  }

  if (status == MPI_STATUS_IGNORE) {
    status = &local_status;
  }
#endif /* TAU_TRACK_MSG */
  

  returnVal = PMPI_Sendrecv( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status );

#ifdef TAU_TRACK_MSG
  if (source != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    PMPI_Get_count( status, MPI_BYTE, &count );
    TAU_TRACE_RECVMSG(status->MPI_TAG, translateRankToWorld(comm, status->MPI_SOURCE), count);
  }
#endif /* TAU_TRACK_MSG */
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
#ifdef TAU_TRACK_MSG
  MPI_Status local_status;
  int size1;
  int typesize2;
#endif /* TAU_TRACK_MSG */

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Sendrecv_replace()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize2 );
    TAU_TRACE_SENDMSG(sendtag, translateRankToWorld(comm, dest), typesize2*count);
    /*
    prof_send( procid_0, dest, sendtag,
               typesize2*count, "MPI_Sendrecv_replace" );
    */
  }

  if (status == MPI_STATUS_IGNORE) {
    status = &local_status;
  }
#endif /* TAU_TRACK_MSG */
  
  returnVal = PMPI_Sendrecv_replace( buf, count, datatype, dest, sendtag, source, recvtag, comm, status );

#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    PMPI_Get_count( status, MPI_BYTE, &size1 );
    TAU_TRACE_RECVMSG(status->MPI_TAG, translateRankToWorld(comm, status->MPI_SOURCE), size1);
    /*
    prof_recv( dest, procid_0, recvtag, size1,
	       "MPI_Sendrecv_replace" );
    */
  }
#endif /* TAU_TRACK_MSG */
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
#ifdef TAU_TRACK_MSG
  int typesize;
#endif /* TAU_TRACK_MSG */

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Ssend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
#ifdef TAU_TRACK_MSG
  if (dest != MPI_PROC_NULL) {
    PMPI_Type_size( datatype, &typesize );
    TAU_TRACE_SENDMSG(tag, translateRankToWorld(comm, dest), typesize*count);
    /*
    prof_send( procid_0, dest, tag, typesize*count,
	       "MPI_Ssend" );
    */
  }
#endif /* TAU_TRACK_MSG */
  
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
  
/* fprintf( stderr, "MPI_Ssend_init call on %d\n", procid_0 ); */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Ssend_init()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Ssend_init( buf, count, datatype, dest, tag, comm, request );

#ifdef TAU_TRACK_MSG
  TauAddRequest(RQ_SEND, count, datatype, dest, tag, comm, request, returnVal, 1);
#endif /* TAU_TRACK_MSG */

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Start( request )
MPI_Request * request;
{
#ifdef TAU_TRACK_MSG
  request_list *rq;
#endif /* TAU_TRACK_MSG */
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Start()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);




#ifdef TAU_TRACK_MSG
  rq = TauGetRequest(*request);
  TauProcessSend(*request, "MPI_Start");
#endif /* TAU_TRACK_MSG */


  returnVal = PMPI_Start( request );


#ifdef TAU_TRACK_MSG
  /* fix up the request since MPI_Start may (will) change it */
  rq->request = *request;
#endif /* TAU_TRACK_MSG */




  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Startall( count, array_of_requests )
int count;
MPI_Request * array_of_requests;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Startall()",  " ", TAU_MESSAGE);
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
#ifdef TAU_TRACK_MSG
  MPI_Request saverequest;
  MPI_Status local_status;
#endif /* TAU_TRACK_MSG */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Test()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
#ifdef TAU_TRACK_MSG
  saverequest = *request;
  if (status == MPI_STATUS_IGNORE) {
    status = &local_status;
  }
#endif /* TAU_TRACK_MSG */

  returnVal = PMPI_Test( request, flag, status );

#ifdef TAU_TRACK_MSG
  if (*flag)
    TauProcessRecv(saverequest, status, "MPI_Test");
#endif /* TAU_TRACK_MSG */

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
#ifdef TAU_TRACK_MSG
  int need_to_free = 0;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];
#endif /* TAU_TRACK_MSG */

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Testall()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
#ifdef TAU_TRACK_MSG
  for (i = 0; i < count; i++) {
    saverequest[i] = array_of_requests[i];
  }
  if (array_of_statuses == MPI_STATUSES_IGNORE) {
    array_of_statuses = (MPI_Status*) malloc (sizeof(MPI_Status)*count);
    need_to_free = 1;
  }
#endif /* TAU_TRACK_MSG */
  
  returnVal = PMPI_Testall( count, array_of_requests, flag, array_of_statuses );

#ifdef TAU_TRACK_MSG
  if (*flag)
  { /* at least one completed */
    for(i=0; i < count; i++)
      TauProcessRecv(saverequest[i], &array_of_statuses[i], "MPI_Testall");
  }
  if (need_to_free) {
    free(array_of_statuses);
  }
#endif /* TAU_TRACK_MSG */

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
#ifdef TAU_TRACK_MSG
  MPI_Status local_status;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];
#endif /* TAU_TRACK_MSG */


  
  TAU_PROFILE_TIMER(tautimer, "MPI_Testany()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_TRACK_MSG
  for (i = 0; i < count; i++) {
    saverequest[i] = array_of_requests[i];
  }
  if (status == MPI_STATUS_IGNORE) {
    status = &local_status;
  }
#endif /* TAU_TRACK_MSG */
  
  returnVal = PMPI_Testany( count, array_of_requests, index, flag, status );


#ifdef TAU_TRACK_MSG
  if (*flag && (*index != MPI_UNDEFINED)) {
    TauProcessRecv(saverequest[*index], status, "MPI_Testany");
  }

#endif /* TAU_TRACK_MSG */
  TAU_PROFILE_STOP(tautimer);
  return returnVal;
}

int  MPI_Test_cancelled( status, flag )
MPI_Status * status;
int * flag;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Test_cancelled()",  " ", TAU_MESSAGE);
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
#ifdef TAU_TRACK_MSG
  int need_to_free = 0;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];
#endif /* TAU_TRACK_MSG */
  
  TAU_PROFILE_TIMER(tautimer, "MPI_Testsome()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_TRACK_MSG
  for (i = 0; i < incount; i++){
    saverequest[i] = array_of_requests[i];
  }
  if (array_of_statuses == MPI_STATUSES_IGNORE) {
    array_of_statuses = (MPI_Status*) malloc (sizeof(MPI_Status)*incount);
    need_to_free = 1;
  }
#endif /* TAU_TRACK_MSG */
  
  returnVal = PMPI_Testsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses );

#ifdef TAU_TRACK_MSG
  for (i=0; i < *outcount; i++) {
    TauProcessRecv( (saverequest [array_of_indices[i]]),
			        &(array_of_statuses[i]),
			        "MPI_Testsome" );
  }
  if (need_to_free) {
    free(array_of_statuses);
  }
#endif /* TAU_TRACK_MSG */

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Type_commit( datatype )
MPI_Datatype * datatype;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_commit()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_contiguous()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_contiguous( count, old_type, newtype );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Type_extent( datatype, extent )
MPI_Datatype datatype;
MPI_Aint * extent;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_extent()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Type_extent( datatype, extent );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Type_free( datatype )
MPI_Datatype * datatype;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_free()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_hindexed()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_hvector()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_indexed()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_lb()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_size()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_struct()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_ub()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Type_vector()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Unpack()",  " ", TAU_MESSAGE);
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

#ifdef TAU_TRACK_MSG
  MPI_Status local_status;
  MPI_Request saverequest;
#endif /* TAU_TRACK_MSG */

  TAU_PROFILE_TIMER(tautimer, "MPI_Wait()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_TRACK_MSG
  if (status == MPI_STATUS_IGNORE) {
    status = &local_status;
  }
#endif /* TAU_TRACK_MSG */


#ifdef TAU_TRACK_MSG
  saverequest = *request;
#endif /* TAU_TRACK_MSG */

  returnVal = PMPI_Wait( request, status );

#ifdef TAU_TRACK_MSG
  TauProcessRecv(saverequest, status, "MPI_Wait");
#endif /* TAU_TRACK_MSG */

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Waitall( count, array_of_requests, array_of_statuses )
int count;
MPI_Request * array_of_requests;
MPI_Status * array_of_statuses;
{
  int  returnVal;
#ifdef TAU_TRACK_MSG
  int need_to_free = 0;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];
#endif /* TAU_TRACK_MSG */

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Waitall()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_TRACK_MSG
  for (i = 0; i < count; i++) {
    saverequest[i] = array_of_requests[i];
  }

  if (array_of_statuses == MPI_STATUSES_IGNORE) {
    array_of_statuses = (MPI_Status*) malloc (sizeof(MPI_Status)*count);
    need_to_free = 1;
  }
#endif /* TAU_TRACK_MSG */
  
  returnVal = PMPI_Waitall( count, array_of_requests, array_of_statuses );

#ifdef TAU_TRACK_MSG
  for(i=0; i < count; i++)
    TauProcessRecv(saverequest[i], &array_of_statuses[i], "MPI_Waitall");

  if (need_to_free) {
    free(array_of_statuses);
  }
#endif /* TAU_TRACK_MSG */

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
#ifdef TAU_TRACK_MSG
  MPI_Status local_status;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];
#endif /* TAU_TRACK_MSG */


  
  TAU_PROFILE_TIMER(tautimer, "MPI_Waitany()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
#ifdef TAU_TRACK_MSG
  for (i = 0; i < count; i++){
    saverequest[i] = array_of_requests[i];
  }
  if (status == MPI_STATUS_IGNORE) {
    status = &local_status;
  }
#endif /* TAU_TRACK_MSG */

  returnVal = PMPI_Waitany( count, array_of_requests, index, status );


#ifdef TAU_TRACK_MSG
  TauProcessRecv( (saverequest[*index]),
			status, "MPI_Waitany" );
#endif /* TAU_TRACK_MSG */

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

#ifdef TAU_TRACK_MSG
  int need_to_free = 0;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];
#endif /* TAU_TRACK_MSG */

  
  TAU_PROFILE_TIMER(tautimer, "MPI_Waitsome()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_TRACK_MSG
  for (i = 0; i < incount; i++) {
    saverequest[i] = array_of_requests[i];
  }

  if (array_of_statuses == MPI_STATUSES_IGNORE) {
    array_of_statuses = (MPI_Status*) malloc (sizeof(MPI_Status)*incount);
    need_to_free = 1;
  }
#endif /* TAU_TRACK_MSG */
  
  returnVal = PMPI_Waitsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses );


#ifdef TAU_TRACK_MSG
  for (i=0; i < *outcount; i++) {
    TauProcessRecv( (saverequest [array_of_indices[i]]),
			        &(array_of_statuses[i]),
			        "MPI_Waitsome" );
  }
  if (need_to_free) {
    free(array_of_statuses);
  }

#endif /* TAU_TRACK_MSG */
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_coords()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_create()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_get()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_map()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_rank()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_shift()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_sub()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Cartdim_get()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Dims_create()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_create()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_get()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_map()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_neighbors()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_neighbors_count()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Graphdims_get()",  " ", TAU_MESSAGE);
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

  TAU_PROFILE_TIMER(tautimer, "MPI_Topo_test()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  
  returnVal = PMPI_Topo_test( comm, top_type );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

/* EOF TauMpi.c */
