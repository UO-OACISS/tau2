/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauMpi.c      				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : MPI Wrapper                                      **
**                                                                         **
****************************************************************************/

#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <TauMetaDataMerge.h>
#include <Profile/TauMon.h>
#include <Profile/TauRequest.h>
#include <Profile/TauSampling.h>
#include <Profile/TauTraceOTF2.h>
#include <Profile/TauUtil.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauPluginInternals.h>
#ifdef TAU_ADIOS
#include "adiost_callback_api.h"
#endif
#include "inttypes.h"

/* Can't include TauMmapMemMgr, becuase it's c++ header.  So declare the
   functions here. */
extern void Tau_MemMgr_finalizeIfNecessary(void);
extern int Tau_get_usesMPI();

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <errno.h>
#include <sys/stat.h>
#include <wordexp.h>

#include <string.h>
#ifdef TAU_BEACON
#include <beacon.h>
#endif /* TAU_BEACON */

#define TAU_MAX_REQUESTS  4096
#ifndef TAU_MAX_MPI_RANKS
#define TAU_MAX_MPI_RANKS 8
#endif /* ifndef */

/* NOTE: We can either track communicator or paths, but not both! */
#ifdef TAU_EXP_TRACK_COMM
#define TAU_TRACK_COMM(c) \
  void *commhandle; \
  commhandle = (void*)c; \
  TAU_PROFILE_PARAM1L((long)commhandle, "comm");
#else
#define TAU_TRACK_COMM(c)
#endif /* TAU_EXP_TRACK_COMM */

#include "check_mpi_version.h"
#include "mpi_tracing_plugin_macros.h"

void TauSyncClocks();
void TauSyncFinalClocks();
int Tau_mergeProfiles_MPI();
void TAUDECL Tau_set_usesMPI(int value);
int TAUDECL tau_totalnodes(int set_or_get, int value);
char * Tau_printRanks(void * comm_ptr);
extern int Tau_signal_initialization();
extern int Tau_mpi_t_initialize();
extern int Tau_mpi_t_cvar_initialize();
extern int Tau_track_mpi_t_here();
extern void Tau_track_mpi_t();
extern int Tau_mpi_t_cleanup();
extern int Tau_msg_send_prolog();
extern int Tau_msg_recv_prolog();
#ifdef TAU_MPI_T_TRACK_GPU_MSGS
#define TAU_MSG_SEND_PROLOG() Tau_msg_send_prolog()
#define TAU_MSG_RECV_PROLOG() Tau_msg_recv_prolog()
#else // TAU_MPI_T_TRACK_GPU_MSGS
#define TAU_MSG_SEND_PROLOG()
#define TAU_MSG_RECV_PROLOG()
#endif

#ifdef TAU_BEACON
extern int TauBeaconSubscribe(char *topic_name, char *topic_scope, void (*handler)(BEACON_receive_topic_t*));
extern void TauBeacon_MPI_T_CVAR_handler(BEACON_receive_topic_t * caught_topic);
#endif /* TAU_BEACON */

/* JCL: Optimized rank translation with cache */
int TauTranslateRankToWorld(MPI_Comm comm, int rank);

#ifndef _AIX
extern void tau_mpi_fortran_init_predefined_constants_(void);
#endif

void tau_mpi_init_predefined_constants()
{
#ifdef TAU_NO_FORTRAN
    TAU_VERBOSE("TAU: WARNING: Not configured with Fortran. You may have trouble with MPI predefined constants like MPI_IN_PLACE\n");
#else
#ifndef _AIX
    tau_mpi_fortran_init_predefined_constants_();
#endif /* _AIX */
#endif
}


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


static int procid_0;

#define track_vector( call, counts, typesize ) { \
    int typesize, commSize, commRank, sendcount = 0, i; \
    PMPI_Comm_rank(comm, &commRank); \
    PMPI_Comm_size(comm, &commSize); \
    if ( commRank == root ) { \
      if (sendtype != MPI_DATATYPE_NULL) { \
        PMPI_Type_size( sendtype, &typesize ); \
      } \
      for (i = 0; i<commSize; i++) { \
	sendcount += counts[i]; \
      } \
      call(typesize*sendcount); \
    } \
  }

static int sum_array (TAU_MPICH3_CONST int *counts, MPI_Datatype type, MPI_Comm comm) {

  int typesize, commSize, commRank, i = 0;
  int total = 0;
  PMPI_Comm_rank(comm, &commRank);
  PMPI_Comm_size(comm, &commSize);
  if (type != MPI_DATATYPE_NULL) {
    PMPI_Type_size(type, &typesize );
  }

  for (i = 0; i<commSize; i++) {
    total += counts[i]; // sum
  }
  return total * typesize;
}

static double* array_stats (TAU_MPICH3_CONST int *counts, MPI_Datatype type, MPI_Comm comm, double vals[5]) {

  int typesize, commSize, commRank, i = 0;
  PMPI_Comm_rank(comm, &commRank);
  PMPI_Comm_size(comm, &commSize);
  if (type != MPI_DATATYPE_NULL) {
    PMPI_Type_size(type, &typesize );
  }
  // check to make sure we have values!
  if (commSize > 0 && counts != NULL) {
    vals[0] = (double)commSize; //count
    vals[1] = (double)counts[0]; //sum
    vals[2] = (double)counts[0]; //min
    vals[3] = (double)counts[0]; //max
    vals[4] = (double)counts[0] * (double)counts[0]; //sumsqr

    for (i = 1; i<commSize; i++) {
      vals[1] += (double)counts[i]; // sum
      vals[2] = (double)counts[i] < vals[2] ? (double)counts[i] : vals[2]; // min
      vals[3] = (double)counts[i] > vals[3] ? (double)counts[i] : vals[3]; // max
      vals[4] += ((double)counts[i] * (double)counts[i]); // sumsqr
    }
    for (i = 1; i<5; i++) {
      vals[i] = vals[i] * (double)typesize;
    }
    vals[1] = vals[1] / (double)commSize; // now it's the mean
  }
  return vals;
}

#define track_allvector( call, counts, typesize ) { \
    int typesize, commSize, commRank, sendcount = 0, i; \
    PMPI_Comm_rank(comm, &commRank); \
    PMPI_Comm_size(comm, &commSize); \
    if(sendtype != MPI_DATATYPE_NULL) { \
        PMPI_Type_size( sendtype, &typesize ); \
    } else { \
	if (recvtype != MPI_DATATYPE_NULL) { \
          PMPI_Type_size( recvtype, &typesize ); \
	} \
    } \
    for (i = 0; i<commSize; i++) { \
      sendcount += counts[i]; \
    } \
    call(typesize*sendcount); \
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







void TauProcessRecv ( request, status, note )
MPI_Request * request;
MPI_Status *status;
char *note;
{
  request_data * rq;
  int otherid, othertag;

  // TAU_PROFILE_TIMER(tautimer, "TauProcessRecv",  " ", TAU_MESSAGE);
  // TAU_PROFILE_START(tautimer);

#ifdef DEBUG
  int myrank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif /* DEBUG */

  rq = TauGetRequestData(request);

  if (!rq) {
#ifdef DEBUG
    fprintf( stderr, "Node %d: Request %lx not found in '%s'.\n",myrank, *request, note );
#endif /* DEBUG */
    // TAU_PROFILE_STOP(tautimer);
    return;                /* request not found */
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
    TAU_TRACE_RECVMSG(othertag, TauTranslateRankToWorld(rq->comm, otherid), rq->size);
    TAU_PLUGIN_RECVMSG(othertag, TauTranslateRankToWorld(rq->comm, otherid), rq->size, 0);
    TAU_WAIT_DATA(rq->size);
  }

  if (!rq->is_persistent) {
    TauDeleteRequestData(request);
  }

  // TAU_PROFILE_STOP(tautimer);
  return;
}

/* This routine traverses the list of requests and checks for RQ_SEND. The
   message is logged if this request matches */


void TauProcessSend ( request, note )
MPI_Request * request;
char *note;
{
  request_data * rq;
  int otherid, othertag;

#ifdef DEBUG
  int myrank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif /* DEBUG */

  rq = TauGetRequestData(request);

  if (!rq) {
#ifdef DEBUG
    fprintf( stderr, "Node %d: Request not found in '%s'.\n",myrank, note );
#endif /* DEBUG */
    return;                /* request not found */
  }
#ifdef DEBUG
  else
  {
    printf("Node %d: Request found %lx\n", myrank, request);
  }
#endif /* DEBUG */

  if ((rq) && rq->status == RQ_SEND)
  {
    otherid = TauTranslateRankToWorld(rq->comm, rq->otherParty);
    othertag = rq->tag;
    /* post the send message */
    TAU_TRACE_SENDMSG(othertag, otherid, rq->size);
    TAU_PLUGIN_SENDMSG(othertag, otherid, rq->size, 0);
  }

  return;
}







/* NOTE: MPI_Type_count was not implemented in mpich-1.2.0. Remove it from this
   list when it is implemented in libpmpich.a */


/* This macro captures the time spent synchronizing at collectives. */
#define TAU_MPI_COLLECTIVE_SYNC(__comm) \
    TAU_PROFILE_TIMER(synctautimer, "MPI Collective Sync", " ", TAU_MESSAGE); \
    TAU_PROFILE_START(synctautimer); \
    PMPI_Barrier(__comm); \
    TAU_PROFILE_STOP(synctautimer);



int   MPI_Allgather( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm )
TAU_MPICH3_CONST void * sendbuf;
int sendcount;
MPI_Datatype sendtype;
void * recvbuf;
int recvcount;
MPI_Datatype recvtype;
MPI_Comm comm;
{
  int   returnVal;
  int   typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allgather()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Allgather( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm );
  if (recvtype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( recvtype, &typesize );
  }
  TAU_ALLGATHER_DATA(typesize*recvcount);

  TIMER_EXIT_COLLECTIVE_EXCH_ALL_EVENT("MPI_Allgather",typesize*sendcount,typesize*recvcount,0,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Allgatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm )
TAU_MPICH3_CONST void * sendbuf;
int sendcount;
MPI_Datatype sendtype;
void * recvbuf;
TAU_MPICH3_CONST int * recvcounts;
TAU_MPICH3_CONST int * displs;
MPI_Datatype recvtype;
MPI_Comm comm;
{
  int   returnVal;
  int   typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allgatherv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Allgatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm );
  if (recvtype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( recvtype, &typesize );
  }

  track_allvector(TAU_ALLGATHER_DATA, recvcounts, typesize);

  if (TAU_DO_TIMER_EXIT) {
    double tmp_array[5] = {0.0};
    TIMER_EXIT_COLLECTIVE_EXCH_V_EVENT("MPI_Allgatherv","sendbytes",sendcount*typesize,array_stats(recvcounts,recvtype,comm,tmp_array),0,comm);
  }
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Allreduce( sendbuf, recvbuf, count, datatype, op, comm )
TAU_MPICH3_CONST void * sendbuf;
void * recvbuf;
int count;
MPI_Datatype datatype;
MPI_Op op;
MPI_Comm comm;
{
  int   returnVal;
  int   typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Allreduce()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Allreduce( sendbuf, recvbuf, count, datatype, op, comm );
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  TAU_ALLREDUCE_DATA(typesize*count);

  TIMER_EXIT_COLLECTIVE_EXCH_EVENT("MPI_Allreduce",typesize*count,0,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Alltoall( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm )
TAU_MPICH3_CONST void * sendbuf;
int sendcount;
MPI_Datatype sendtype;
void * recvbuf;
int recvcnt;
MPI_Datatype recvtype;
MPI_Comm comm;
{
  int  returnVal;
  int   typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Alltoall()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);

  returnVal = PMPI_Alltoall( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm );
  if (sendtype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( sendtype, &typesize );
  }
  TAU_ALLTOALL_DATA(typesize*sendcount);

  TIMER_EXIT_COLLECTIVE_EXCH_ALL_EVENT("MPI_Alltoall",typesize*sendcount,typesize*recvcnt,0,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Alltoallv( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm )
TAU_MPICH3_CONST void * sendbuf;
TAU_MPICH3_CONST int * sendcnts;
TAU_MPICH3_CONST int * sdispls;
MPI_Datatype sendtype;
void * recvbuf;
TAU_MPICH3_CONST int * recvcnts;
TAU_MPICH3_CONST int * rdispls;
MPI_Datatype recvtype;
MPI_Comm comm;
{
  int   returnVal;
  int tracksize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Alltoallv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Alltoallv( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm );

  tracksize = sum_array(sendcnts, sendtype, comm);
  tracksize += sum_array(recvcnts, recvtype, comm);

  TAU_ALLTOALL_DATA(tracksize);

  if (TAU_DO_TIMER_EXIT) {
    double tmp_array1[5] = {0.0};
    double tmp_array2[5] = {0.0};
    TIMER_EXIT_COLLECTIVE_EXCH_AAV_EVENT("MPI_Alltoallv",array_stats(sendcnts,sendtype,comm,tmp_array1),array_stats(recvcnts,recvtype,comm,tmp_array2),comm);
  }
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Barrier( comm )
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Barrier()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Barrier( comm );

  TIMER_EXIT_COLLECTIVE_SYNC_EVENT("MPI_Barrier",comm);
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
  int   typesize = 0;
#ifdef TAU_MPI_BCAST_HISTOGRAM
  TAU_REGISTER_CONTEXT_EVENT(c1, "Message size in MPI_Bcast [0, 1KB)");
  TAU_REGISTER_CONTEXT_EVENT(c2, "Message size in MPI_Bcast [1KB, 10KB)");
  TAU_REGISTER_CONTEXT_EVENT(c3, "Message size in MPI_Bcast [10KB, 100KB)");
  TAU_REGISTER_CONTEXT_EVENT(c4, "Message size in MPI_Bcast [100KB, 1000KB)");
  TAU_REGISTER_CONTEXT_EVENT(c5, "Message size in MPI_Bcast [1000KB+]");

#endif

  TAU_PROFILE_TIMER(tautimer, "MPI_Bcast()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);

  returnVal = PMPI_Bcast( buffer, count, datatype, root, comm );
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }

#ifdef TAU_MPI_BCAST_HISTOGRAM
  unsigned long long volume = typesize * count;
  if (volume  < 1024) {
    TAU_CONTEXT_EVENT(c1, volume);
  } else {
    if (volume < 10240) {
      TAU_CONTEXT_EVENT(c2, volume);
    } else {
      if (volume < 102400) {
        TAU_CONTEXT_EVENT(c3, volume);
      } else {
        if (volume < 1024000) {
            TAU_CONTEXT_EVENT(c4, volume);
        } else {
          TAU_CONTEXT_EVENT(c5, volume);
        }
      }
    }
  }
#endif

  TAU_BCAST_DATA(typesize*count);

  TIMER_EXIT_COLLECTIVE_EXCH_EVENT("MPI_Bcast",typesize*count,root,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Gather( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm )
TAU_MPICH3_CONST void * sendbuf;
int sendcnt;
MPI_Datatype sendtype;
void * recvbuf;
int recvcount;
MPI_Datatype recvtype;
int root;
MPI_Comm comm;
{
  int   returnVal;
  int   typesize = 0;
  int   rank;

  TAU_PROFILE_TIMER(tautimer, "MPI_Gather()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Gather( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm );

  PMPI_Comm_rank ( comm, &rank );
  if (recvtype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( recvtype, &typesize );
  }
  if (rank == root) {
    TAU_GATHER_DATA(typesize*recvcount);
  }

  TIMER_EXIT_COLLECTIVE_EXCH_ALL_EVENT("MPI_Gather",typesize*sendcnt,typesize*recvcount,root,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Gatherv( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm )
TAU_MPICH3_CONST void * sendbuf;
int sendcnt;
MPI_Datatype sendtype;
void * recvbuf;
TAU_MPICH3_CONST int * recvcnts;
TAU_MPICH3_CONST int * displs;
MPI_Datatype recvtype;
int root;
MPI_Comm comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Gatherv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Gatherv( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm );

  track_vector(TAU_GATHER_DATA, recvcnts, recvtype);

  if (TAU_DO_TIMER_EXIT) {
    double tmp_array[5] = {0.0};
    int typesize = 0;
    if (sendtype != MPI_DATATYPE_NULL) {
      PMPI_Type_size( sendtype, &typesize );
    }
    TIMER_EXIT_COLLECTIVE_EXCH_V_EVENT("MPI_Gatherv","sendbytes",sendcnt*typesize,array_stats(recvcnts,recvtype,comm,tmp_array),root,comm);
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
TAU_MPICH3_CONST void * sendbuf;
void * recvbuf;
TAU_MPICH3_CONST int * recvcnts;
MPI_Datatype datatype;
MPI_Op op;
MPI_Comm comm;
{
  int   returnVal;
  int   typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Reduce_scatter()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Reduce_scatter( sendbuf, recvbuf, recvcnts, datatype, op, comm );
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  TAU_REDUCESCATTER_DATA(typesize*(*recvcnts));

  TIMER_EXIT_COLLECTIVE_EXCH_EVENT("MPI_Reduce_scatter",typesize*(*recvcnts),0,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Reduce( sendbuf, recvbuf, count, datatype, op, root, comm )
TAU_MPICH3_CONST void * sendbuf;
void * recvbuf;
int count;
MPI_Datatype datatype;
MPI_Op op;
int root;
MPI_Comm comm;
{
  int   returnVal;
  int   typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Reduce()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Reduce( sendbuf, recvbuf, count, datatype, op, root, comm );
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  TAU_REDUCE_DATA(typesize*count);

  TIMER_EXIT_COLLECTIVE_EXCH_EVENT("MPI_Reduce",typesize*count,root,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Scan( sendbuf, recvbuf, count, datatype, op, comm )
TAU_MPICH3_CONST void * sendbuf;
void * recvbuf;
int count;
MPI_Datatype datatype;
MPI_Op op;
MPI_Comm comm;
{
  int   returnVal;
  int   typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scan()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Scan( sendbuf, recvbuf, count, datatype, op, comm );
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  TAU_SCAN_DATA(typesize*count);

  TIMER_EXIT_COLLECTIVE_EXCH_EVENT("MPI_Scan",typesize*count,0,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Scatter( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm )
TAU_MPICH3_CONST void * sendbuf;
int sendcnt;
MPI_Datatype sendtype;
void * recvbuf;
int recvcnt;
MPI_Datatype recvtype;
int root;
MPI_Comm comm;
{
  int   returnVal;
  int   typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scatter()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Scatter( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm );
  if (sendtype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( sendtype, &typesize );
  }
  TAU_SCATTER_DATA(typesize*sendcnt);

  TIMER_EXIT_COLLECTIVE_EXCH_ALL_EVENT("MPI_Scatter",typesize*sendcnt,typesize*recvcnt,root,comm);
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Scatterv( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm )
TAU_MPICH3_CONST void * sendbuf;
TAU_MPICH3_CONST int * sendcnts;
TAU_MPICH3_CONST int * displs;
MPI_Datatype sendtype;
void * recvbuf;
int recvcnt;
MPI_Datatype recvtype;
int root;
MPI_Comm comm;
{
  int returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Scatterv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Scatterv( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm );

  track_vector(TAU_SCATTER_DATA, sendcnts, typesize);

  if (TAU_DO_TIMER_EXIT) {
    double tmp_array[5] = {0.0};
    int typesize = 0;
    if (sendtype != MPI_DATATYPE_NULL) {
      PMPI_Type_size( sendtype, &typesize );
    }
    TIMER_EXIT_COLLECTIVE_EXCH_V_EVENT("MPI_Scatterv","recvbytes",recvcnt*typesize,array_stats(sendcnts,recvtype,comm,tmp_array),root,comm);
  }
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

// OpenMPI 4 and later have removed some functions deleted in MPI 3.0
// #if !defined(OMPI_MAJOR_VERSION) || (OMPI_MAJOR_VERSION < 4)
#if MPI_VERSION < 2

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

#ifdef TAU_ENABLE_MPI_ATTR_GET
int   MPI_Attr_get( comm, keyval, attr_value, flag )
MPI_Comm comm;
int keyval;
void * attr_value;
int * flag;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Attr_get()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Attr_get( comm, keyval, attr_value, flag );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

#endif /* TAU_ENABLE_MPI_ATTR_GET */

int   MPI_Attr_put( comm, keyval, attr_value )
MPI_Comm comm;
int keyval;
void * attr_value;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Attr_put()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Attr_put( comm, keyval, attr_value );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

#else /* MPI_VERSION < 2 */

/* replacements defined in TauMpiExtensions.c, for some reason */

#endif /* MPI_VERSION < 2 */

#ifndef TAU_MPI_DISABLE_COMM_WRAPPERS
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

int Tau_setupCommunicatorInfo(MPI_Comm * comm)  {
  return 0;
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
  TIMER_EXIT_COMM_CREATE_EVENT(comm, group, *comm_out);

  Tau_setupCommunicatorInfo(comm_out);
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
  TAU_MPI_COLLECTIVE_SYNC(comm);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Comm_dup( comm, comm_out );
  TIMER_EXIT_COMM_DUP_EVENT(comm, *comm_out);

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Comm_free( comm )
MPI_Comm * comm;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_free()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  //TAU_MPI_COLLECTIVE_SYNC(comm);

  MPI_Comm silly =  *comm;
  returnVal = PMPI_Comm_free( &(silly) );
  TIMER_EXIT_COMM_FREE_EVENT(*comm);

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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Comm_group( comm, group );
  TIMER_EXIT_COMM_GROUP_EVENT(comm,*group);

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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Comm_rank( comm, rank );

  TAU_PROFILE_STOP(tautimer);

  /* Set the node as we did in MPI_Init */
  if (comm == MPI_COMM_WORLD) {
    TAU_PROFILE_SET_NODE(*rank);
    Tau_set_usesMPI(1);
  }

  return returnVal;
}

int   MPI_Comm_remote_group( comm, group )
MPI_Comm comm;
MPI_Group * group;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_remote_group()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  TAU_TRACK_COMM(comm);
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

  TAU_TRACK_COMM(comm);
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

/**************************************************************************
 Experimental routine to track communicator splits in metadata
 This will create a metadata item such as:
   Name : MPI_Comm 102140608
   Value : 0 2 4 6 ...
***************************************************************************/
#ifdef TAU_EXP_TRACK_COMM
void tau_exp_track_comm_split (MPI_Comm oldcomm, MPI_Comm newcomm) {
  int worldrank;
  int newCommSize;
  void *oldcommhandle, *newcommhandle;
  int i;
  char buffer[16384];
  char catbuffer[2048];
  char namebuffer[512];
  int limit;

  oldcommhandle = (void*)oldcomm;
  newcommhandle = (void*)newcomm;

/*   printf ("comm %p split into %p for %d\n", oldcommhandle, newcommhandle, procid_0); */
  MPI_Comm_size(newcomm, &newCommSize);
/*   printf ("comm %p split into %p for %d, new size = %d\n", oldcommhandle, newcommhandle, procid_0, newCommSize); */

  /* initialize to empty */
  buffer[0] = 0;

  limit = (newCommSize < TAU_MAX_MPI_RANKS) ? newCommSize : TAU_MAX_MPI_RANKS;
  for (i=0; i<limit; i++) {
    worldrank = TauTranslateRankToWorld(newcomm, i);
/*     printf ("comm %p has world member %d\n", newcommhandle, worldrank); */
    sprintf (catbuffer, "%d ", worldrank);
    strcat(buffer, catbuffer);
  }
  if (limit < newCommSize) {
    strcat(buffer, " ...");
  }

/*   printf ("buffer is %s\n", buffer); */
  sprintf (namebuffer, "MPI_Comm %p", newcommhandle);
  TAU_METADATA(namebuffer, buffer);
}
#endif /* TAU_EXP_TRACK_COMM */

int   MPI_Comm_split( comm, color, key, comm_out )
MPI_Comm comm;
int color;
int key;
MPI_Comm * comm_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_split()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  //TAU_MPI_COLLECTIVE_SYNC(comm);

  MPI_Comm newcomm = comm;
  returnVal = PMPI_Comm_split( newcomm, color, key, comm_out );
  TIMER_EXIT_COMM_SPLIT_EVENT(comm,color,key,*comm_out);

#ifdef TAU_EXP_TRACK_COMM
  tau_exp_track_comm_split(newcomm, *comm_out);
#endif /* TAU_EXP_TRACK_COMM */

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

void Tau_handle_comm_spawn(MPI_Comm comm, MPI_Comm intercomm) {
    static int tau_comm_spawn_num = 0;
    tau_comm_spawn_num++;
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    // Send the generation number to the spawned processes, which
    // will join the Bcast in MPI_Init
    if(comm_rank == 0) {
        PMPI_Bcast(&tau_comm_spawn_num, 1, MPI_INT, MPI_ROOT, intercomm);
    } else {
        PMPI_Bcast(&tau_comm_spawn_num, 1, MPI_INT, MPI_PROC_NULL, intercomm);
    }
}

int MPI_Comm_spawn(TAU_NONMPC_CONST char *command, char *argv[], int maxprocs,
    MPI_Info info, int root, MPI_Comm comm,
    MPI_Comm *intercomm, int array_of_errcodes[]) {
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_spawn()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  //TAU_MPI_COLLECTIVE_SYNC(comm);

  const char * tau_exec_args = TauEnv_get_tau_exec_args();
  TAU_NONMPC_CONST char * tau_exec_path = (TAU_NONMPC_CONST char *)TauEnv_get_tau_exec_path();
  int allocated_argv = 0;
  wordexp_t p;
  if(tau_exec_args != NULL && tau_exec_args[0] != '\0') {
    // This program was launched through tau_exec
    const char * old_command = command;
    char ** old_argv = argv;
    size_t old_argc = 0;
    if(old_argv != NULL) {
        char * arg;
        for(arg = old_argv[old_argc]; arg != NULL; arg = old_argv[++old_argc]);
    }
    wordexp(tau_exec_args, &p, WRDE_NOCMD);
    argv = malloc((p.we_wordc + old_argc + 2) * sizeof(char*));
    size_t offset;
    for(offset = 0; offset < p.we_wordc; ++offset) {
      argv[offset] = p.we_wordv[offset];
    }
    argv[offset++] = (char*)old_command;
    size_t i;
    for(i = 0; i < old_argc; ++i) {
      argv[offset++] = old_argv[i];
    }
    argv[offset] = NULL;

    command = tau_exec_path;
    allocated_argv = 1;

  }

  returnVal = PMPI_Comm_spawn(command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes);
  Tau_handle_comm_spawn(comm, *intercomm);

  if(allocated_argv == 1) {
    free(argv);
    wordfree(&p);
  }

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int MPI_Comm_spawn_multiple(int count, char *array_of_commands[],
    char **array_of_argv[], TAU_NONMPC_CONST int array_of_maxprocs[], TAU_NONMPC_CONST MPI_Info
    array_of_info[], int root, MPI_Comm comm, MPI_Comm *intercomm,
    int array_of_errcodes[]) {
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Comm_spawn_multiple()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Comm_spawn_multiple(count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes);
  Tau_handle_comm_spawn(comm, *intercomm);

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

  TAU_TRACK_COMM(comm);
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
  TIMER_EXIT_GROUP_DIFFERENCE_EVENT(group1, group2, *group_out);

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Group_excl( group, n, ranks, newgroup )
MPI_Group group;
int n;
TAU_MPICH3_CONST int * ranks;
MPI_Group * newgroup;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_excl()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Group_excl( group, n, ranks, newgroup );
  TIMER_EXIT_GROUP_EXCL_EVENT(group,n,ranks,*newgroup);

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
TAU_MPICH3_CONST int * ranks;
MPI_Group * group_out;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_incl()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Group_incl( group, n, ranks, group_out );
  TIMER_EXIT_GROUP_INCL_EVENT(group,n,ranks,*group_out);

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
  TIMER_EXIT_GROUP_INTERSECTION_EVENT(group1, group2, *group_out);

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
  TIMER_EXIT_GROUP_RANGE_EXCL_EVENT(group, n, ranges, *newgroup);

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

#ifdef TAU_MPI_GROUP_RANGE_INCL_DEFINED
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
  TIMER_EXIT_GROUP_RANGE_INCL_EVENT(group, n, ranges, *newgroup);

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
#endif /* TAU_MPI_GROUP_RANGE_INCL_DEFINED */

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
TAU_MPICH3_CONST int * ranks_a;
MPI_Group group_b;
int * ranks_b;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Group_translate_ranks()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Group_translate_ranks( group_a, n, ranks_a, group_b, ranks_b );
  TIMER_EXIT_GROUP_TRANSLATE_RANKS_EVENT(group_a, n, ranks_a, group_b, ranks_b);

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
  TIMER_EXIT_GROUP_UNION_EVENT(group1, group2, *group_out);

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

  TAU_TRACK_COMM(local_comm);
  returnVal = PMPI_Intercomm_create( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out );
  TIMER_EXIT_INTERCOMM_CREATE_EVENT(local_comm, local_leader, peer_comm, remote_leader, tag, *comm_out);

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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Intercomm_merge( comm, high, comm_out );
  TIMER_EXIT_INTERCOMM_MERGE_EVENT(comm, high, *comm_out);

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
#endif // TAU_MPI_DISABLE_COMM_WRAPPERS

// OpenMPI 4 and later have removed some functions deleted in MPI 3.0
// #if !defined(OMPI_MAJOR_VERSION) || (OMPI_MAJOR_VERSION < 4)
#if MPI_VERSION < 2

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

#else /* MPI_VERSION < 2 */

/* replacements defined in TauMpiExtensions.c, for some reason */

#endif /* MPI_VERSION >= 2 */

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

#ifndef TAU_WINDOWS
  if (TauEnv_get_track_signals()) {
    kill(getpid(), SIGABRT);
  }
#endif
  TAU_TRACK_COMM(comm);
  TAU_PROFILE_EXIT("MPI_Abort");
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

// OpenMPI 4 and later have removed some functions deleted in MPI 3.0
//#if !defined(OMPI_MAJOR_VERSION) || (OMPI_MAJOR_VERSION < 4)
#if MPI_VERSION < 2

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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Errhandler_get( comm, errhandler );

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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Errhandler_set( comm, errhandler );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

#else  /* MPI_VERSION > 2 */

/* replacements defined in TauMpiExtensions.c, for some reason */

#endif /* MPI_VERSION < 2 */

int  MPI_Errhandler_free( MPI_Errhandler * errhandler ) {
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Errhandler_free()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Errhandler_free( errhandler );

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

// OpenMPI 4 and later have removed some functions deleted in MPI 3.0
#if !defined(OMPI_MAJOR_VERSION) || (OMPI_MAJOR_VERSION < 4)
#endif //OMPI_MAJOR_VERSION

#if !defined(TAU_MPC)
int tau_mpi_finalized = 0;
int TAU_MPI_Finalized() {
  //fprintf(stdout, "In TAU_MPI_Finalized(): tau_mpi_finalized=%d\n", tau_mpi_finalized);
  return tau_mpi_finalized;
}
#endif

int TauEnv_get_track_memory_footprint(void);

void finalizeCallSites_if_necessary();
int  MPI_Finalize(  )
{
  int  returnVal;
  char procname[MPI_MAX_PROCESSOR_NAME];
  int  procnamelength;

  TAU_VERBOSE("TAU: Call MPI_Finalize()\n");

  Tau_flush_gpu_activity();

  TAU_PROFILE_TIMER(tautimer, "MPI_Finalize()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef TAU_MPI_T
  Tau_track_mpi_t_here();

  /*Clean up and finalize the MPI_T interface*/
  Tau_mpi_t_cleanup();

  returnVal = PMPI_T_finalize();
  if (returnVal != MPI_SUCCESS) {
    printf("TAU: Call to MPI_T_finalize failed\n");
  }

#endif /* TAU_MPI_T */

#ifdef TAU_SOS
  //TIMER_EXIT_stop_worker();
#endif

  if (TauEnv_get_synchronize_clocks()) {
    TauSyncFinalClocks();
  }

  PMPI_Get_processor_name(procname, &procnamelength);
  TAU_METADATA("MPI Processor Name", procname);

  if (Tau_get_node() < 0) {
    /* Grab the node id, we don't always wrap mpi_init */
    PMPI_Comm_rank( MPI_COMM_WORLD, &procid_0 );
    TAU_PROFILE_SET_NODE(procid_0 );
    Tau_set_usesMPI(1);
  }

#ifdef TAU_BGP
  /* BGP counters */
  int numCounters, mode, upcErr;
  x_uint64 counterVals[1024];

  if (TauEnv_get_ibm_bg_hwp_counters()) {
    PMPI_Barrier(MPI_COMM_WORLD);
    Tau_Bg_hwp_counters_stop(&numCounters, counterVals, &mode, &upcErr);
    if (upcErr != 0) {
      printf("  ** Error stopping UPC performance counters");
    }

    Tau_Bg_hwp_counters_output(&numCounters, counterVals, &mode, &upcErr);
  }
#endif /* TAU_BGP */

#ifndef TAU_WINDOWS
#ifndef _AIX
  /* Shutdown EBS after Finalize to allow Profiles to be written out
     correctly. Also allows profile merging (or unification) to be
     done correctly. */
  if (TauEnv_get_callsite()) {
    finalizeCallSites_if_necessary();
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

  Tau_MemMgr_finalizeIfNecessary();

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    //    Tau_sampling_finalizeNode();

    Tau_sampling_finalize_if_necessary(Tau_get_local_tid());
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

  /* *CWL* This might be generalized to perform a final monitoring dump.
     For now, we should let merging handle the data.
#ifdef TAU_MON_MPI
    Tau_collate_writeProfile();
#else
  */

  // merge TAU metadata
  if (TauEnv_get_merge_metadata()) {
    Tau_metadataMerge_mergeMetaData();
  }

  /* Create a merged profile if requested */
  if (TauEnv_get_profile_format() == TAU_FORMAT_MERGED) {
    /* *CWL* - properly record intermediate values (the same way snapshots work).
               Note that we do not want to shut down the timers as yet. There is
	       still potentially life after MPI_Finalize where TAU is concerned.
     */
    /* KAH - NO! this is the wrong time to do this. THis is also done in the
     * snapshot writer. If you do it twice, you get double values for main... */
    //TauProfiler_updateAllIntermediateStatistics();
    Tau_mergeProfiles_MPI();
  }

#ifdef TAU_MONITORING
  Tau_mon_disconnect();
#endif /* TAU_MONITORING */

  /*Invoke plugins only if both plugin path and plugins are specified
   *Do this first, because the plugin can write TAU_METADATA as recommendations to the user*/
  if(Tau_plugins_enabled.pre_end_of_execution) {
    Tau_plugin_event_pre_end_of_execution_data_t plugin_data;
    plugin_data.tid = Tau_get_local_tid();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_PRE_END_OF_EXECUTION, "*", &plugin_data);
  }

#ifdef TAU_OTF2
   if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
     TauTraceOTF2ShutdownComms(Tau_get_local_tid());
   }
#endif

  if (TauEnv_get_track_memory_footprint()) {
    TAU_TRACK_MEMORY_FOOTPRINT_HERE();
  }

  returnVal = PMPI_Finalize();

  TAU_PROFILE_STOP(tautimer);

  Tau_stop_top_level_timer_if_necessary();
#ifndef TAU_MPC
  tau_mpi_finalized = 1;
#endif /* TAU_MPC */

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

int Tau_MPI_T_initialization(void) {
#ifdef TAU_MPI_T
  int returnVal = 0;
  char *pycoolr_cvar_topic_name = "MPI_T_CVARS";
  char *pycoolr_cvar_topic_scope = "global";

  if (TauEnv_get_track_mpi_t_pvars()) {
    Tau_mpi_t_initialize();
  }

  returnVal = Tau_mpi_t_cvar_initialize();

  #ifdef TAU_BEACON
  if(getenv("BEACON_TOPOLOGY_SERVER_ADDR") != NULL) {
    returnVal = TauBeaconSubscribe(pycoolr_cvar_topic_name, pycoolr_cvar_topic_scope, TauBeacon_MPI_T_CVAR_handler);
  }  else {
    fprintf(stderr,"TAU: Warning! TAU is built with beacon but BEACON_TOPOLOGY_SERVER_ADDR is not set. Carrying on without enabling beacon support\n");
    returnVal = 1;
  }
  #endif /* TAU_BEACON */

  return returnVal;
#else
  return 0;
#endif /* TAU_MPI_T */
}

int mkdirp(const char *path) {
    const size_t len = strlen(path);
    char _path[4096];
    char *p;

    errno = 0;

    /* Copy string so its mutable */
    if (len > sizeof(_path)-1) {
            errno = ENAMETOOLONG;
            return -1;
        }
    strcpy(_path, path);

    /* Iterate the string */
    for (p = _path + 1; *p; p++) {
            if (*p == '/') {
                        /* Temporarily truncate */
                        *p = '\0';

                        if (mkdir(_path, S_IRWXU) != 0) {
                                        if (errno != EEXIST)
                                            return -1;
                                    }

                        *p = '/';
                    }
        }

    if (mkdir(_path, S_IRWXU) != 0) {
            if (errno != EEXIST)
                return -1;
        }

    return 0;
}

void Tau_handle_spawned_init(MPI_Comm intercomm) {
  int generation_num;
  PMPI_Bcast(&generation_num, 1, MPI_INT, 0, intercomm);
  const char * profiledir = TauEnv_get_profiledir();
  const char * tracedir = TauEnv_get_profiledir();
  char new_profiledir[4096];
  char new_tracedir[4096];
  snprintf(new_profiledir, 4096, "%s/spawn-%d", profiledir, generation_num);
  snprintf(new_tracedir, 4096, "%s/spawn-%d", tracedir, generation_num);
  mkdirp(new_profiledir);
  mkdirp(new_tracedir);
  TauEnv_set_profiledir(new_profiledir);
  TauEnv_set_tracedir(new_tracedir);
  TAU_VERBOSE("TAU_INIT: MPI_Comm_spawn generation %d\n", generation_num);
}

int  MPI_Init( argc, argv )
int * argc;
char *** argv;
{
  int  returnVal;
  int  size;
  char procname[MPI_MAX_PROCESSOR_NAME];
  int  procnamelength;

  // prevent double-initialization for SHMEM cases
  int init_flag = 0;
  MPI_Initialized(&init_flag);
  if(init_flag == 0) {

    TAU_PROFILE_TIMER(tautimer, "MPI_Init()",  " ", TAU_MESSAGE);
    Tau_create_top_level_timer_if_necessary();
    TAU_PROFILE_START(tautimer);

    tau_mpi_init_predefined_constants();

#ifdef TAU_ADIOS
    // this is only here to force the linker to resolve the adiost_tool symbol
    // before the weak one in the ADIOS static library gets pulled in, and prevents
    // TAU from replacing it.
    adiost_tool();
#endif

    Tau_disable_pthread_tracking();

#ifdef TAU_ADIOS2
    int provided;
    TAU_VERBOSE("%s Initializing with PMPI_Init_thread\n", __func__);
    returnVal = PMPI_Init_thread( argc, argv, MPI_THREAD_MULTIPLE, &provided );
    if (provided != MPI_THREAD_MULTIPLE && provided != MPI_THREAD_FUNNELED) {
      fprintf(stderr, "ERROR!!!  MPI implementation doesn't provide threaded support.\nADIOS2 output from TAU likely won't work.\n");
    }
#else
    TAU_VERBOSE("%s Initializing with PMPI_Init\n", __func__);
    returnVal = PMPI_Init( argc, argv );
#endif

    TAU_PROFILE_STOP(tautimer);

    PMPI_Comm_rank( MPI_COMM_WORLD, &procid_0 );
    TAU_PROFILE_SET_NODE(procid_0 );
    Tau_set_usesMPI(1);

    PMPI_Comm_size( MPI_COMM_WORLD, &size );
    tau_totalnodes(1, size); /* Set the totalnodes */

    PMPI_Get_processor_name(procname, &procnamelength);
    TAU_METADATA("MPI Processor Name", procname);

    Tau_enable_pthread_tracking();

#ifdef TAU_MPI_T
    Tau_MPI_T_initialization();
    Tau_track_mpi_t();
#endif /* TAU_MPI_T */

    MPI_Comm parent;
    PMPI_Comm_get_parent(&parent);
    if(parent != MPI_COMM_NULL) {
      // This process was created through MPI_Comm_spawn
      Tau_handle_spawned_init(parent);
    }

#ifndef TAU_WINDOWS
#ifndef _AIX
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_init_if_necessary();
    }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

    /* Initialize the plugin system */
    Tau_initialize_plugin_system();

    Tau_signal_initialization();

#ifdef TAU_MONITORING
    Tau_mon_connect();
#endif /* TAU_MONITORING */

#ifdef TAU_BGP
    if (TauEnv_get_ibm_bg_hwp_counters()) {
      int upcErr;
      Tau_Bg_hwp_counters_start(&upcErr);
      if (upcErr != 0) {
        printf("TAU ERROR: ** Error starting IBM BGP UPC hardware performance counters\n");
      }
      PMPI_Barrier(MPI_COMM_WORLD);
    }
#endif /* TAU_BGP */
    if (TauEnv_get_synchronize_clocks()) {
      TauSyncClocks();
    }
  } else {
    returnVal = MPI_SUCCESS;
    Tau_set_usesMPI(1);
  }

  writeMetaDataAfterMPI_Init();

  Tau_post_init();

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_init_if_necessary();
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

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

  TAU_VERBOSE("call TAU MPI_Init_thread()\n");

  TAU_PROFILE_TIMER(tautimer, "MPI_Init_thread()",  " ", TAU_MESSAGE);
  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_START(tautimer);

  tau_mpi_init_predefined_constants();

  returnVal = PMPI_Init_thread( argc, argv, required, provided );

  TAU_PROFILE_STOP(tautimer);

  PMPI_Comm_rank( MPI_COMM_WORLD, &procid_0 );
  TAU_PROFILE_SET_NODE(procid_0 );
  Tau_set_usesMPI(1);

  PMPI_Comm_size( MPI_COMM_WORLD, &size );
  tau_totalnodes(1, size); /* Set the totalnodes */

  PMPI_Get_processor_name(procname, &procnamelength);
  TAU_METADATA("MPI Processor Name", procname);

#ifdef TAU_MPI_T
  returnVal = Tau_MPI_T_initialization();
  Tau_track_mpi_t();
#endif /* TAU_MPI_T */

  MPI_Comm parent;
  MPI_Comm_get_parent(&parent);
  if(parent != MPI_COMM_NULL) {
    // This process was created through MPI_Comm_spawn
    Tau_handle_spawned_init(parent);
  }

  /* Initialize the plugin system */
  Tau_initialize_plugin_system();

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_init_if_necessary();
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

  Tau_signal_initialization();

#ifdef TAU_BGP
  if (TauEnv_get_ibm_bg_hwp_counters()) {
    int upcErr;
    Tau_Bg_hwp_counters_start(&upcErr);
    if (upcErr != 0) {
      printf("TAU ERROR: ** Error starting IBM BGP UPC hardware performance counters\n");
    }
    PMPI_Barrier(MPI_COMM_WORLD);
  }
#endif /* TAU_BGP */

  if (TauEnv_get_synchronize_clocks()) {
    TauSyncClocks();
    //TauSyncClocks takes no arguments.
    //TauSyncClocks(procid_0, size);
  }

  writeMetaDataAfterMPI_Init();

  Tau_post_init();

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_init_if_necessary();
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

  return returnVal;
}
#endif /* TAU_MPI_THREADED */


#if 0
int  MPI_Initialized( flag )
int * flag;
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Initialized()",  " ", TAU_MESSAGE);
  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Initialized( flag );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
#endif


#ifdef TAU_ENABLE_MPI_WTIME
double  MPI_Wtick(  )
{
  double  returnVal;

  /* To enable the instrumentation change group to TAU_MESSAGE */
  TAU_PROFILE_TIMER(tautimer, "MPI_Wtick()",  " ", TAU_DISABLE);
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

#ifdef TAU_ENABLE_MPI_GET_VERSION
int MPI_Get_version( int *version, int *subversion )
{
  int  returnVal;

  /* To enable the instrumentation change group to TAU_MESSAGE */
  TAU_PROFILE_TIMER(tautimer, "MPI_Get_version()",  " ", TAU_DISABLE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Get_version( version, subversion );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
#endif

// OpenMPI 4 and later have removed some functions deleted in MPI 3.0
// #if !defined(OMPI_MAJOR_VERSION) || (OMPI_MAJOR_VERSION < 4)
#if MPI_VERSION < 2

#if (defined(TAU_SGI_MPT_MPI) || defined(TAU_NEC_SX) || defined(TAU_NEC_MPI_VH_SX))
int  MPI_Address( void * location, void * address )
#else
int  MPI_Address( const void * location, MPI_Aint * address)
#endif /* TAU_SGI_MPT_MPI */
{
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Address()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Address( location, address );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

#else /* MPI_VERSION > 2 */

/* replacements defined in TauMpiExtensions.c, for some reason */

#endif /* MPI_VERSION < 2 */

int  MPI_Bsend( buf, count, datatype, dest, tag, comm )
TAU_MPICH3_CONST void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
{
  int  returnVal;
  int typesize = 0;


  TAU_PROFILE_TIMER(tautimer, "MPI_Bsend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), typesize*count);
    }
  }
  TAU_PLUGIN_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), typesize*count, 0);
  TAU_TRACK_COMM(comm);

  returnVal = PMPI_Bsend( buf, count, datatype, dest, tag, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Bsend_init( buf, count, datatype, dest, tag, comm, request )
TAU_MPICH3_CONST void * buf;
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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Bsend_init( buf, count, datatype, dest, tag, comm, request );

  if (TauEnv_get_track_message()) {
    TauAddRequestData(RQ_SEND, count, datatype, dest, tag, comm, request, returnVal, 1);
  }

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

  if (TauEnv_get_track_message()) {
    TauDeleteRequestData(request);
  }

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

  if (TauEnv_get_track_message()) {
    TauDeleteRequestData(request);
  }

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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Recv_init( buf, count, datatype, source, tag, comm, request );

  TAU_PROFILE_STOP(tautimer);

  if (TauEnv_get_track_message()) {
    TauAddRequestData(RQ_RECV, count, datatype, source, tag, comm, request, returnVal, 1);
  }
  return returnVal;
}

int  MPI_Send_init( buf, count, datatype, dest, tag, comm, request )
TAU_MPICH3_CONST void * buf;
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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Send_init( buf, count, datatype, dest, tag, comm, request );

  /* we need to store the request and associate it with the size/tag so MPI_Start can
     retrieve it and log the TAU_TRACE_SENDMSG */
if (TauEnv_get_track_message()) {
  TauAddRequestData(RQ_SEND, count, datatype, dest, tag, comm, request, returnVal, 1);
}

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Get_elements( status, datatype, elements )
TAU_MPICH3_CONST MPI_Status * status;
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
TAU_MPICH3_CONST MPI_Status * status;
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
TAU_MPICH3_CONST void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;
  int typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Ibsend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), count * typesize);
    }
  }
  TAU_PLUGIN_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), count * typesize, 0);

  TAU_TRACK_COMM(comm);
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

  TAU_TRACK_COMM(comm);
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

#ifdef DEBUG
  int myrank;
#endif /* DEBUG */

  TAU_PROFILE_TIMER(tautimer, "MPI_Irecv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

#ifdef DEBUG
  PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif /* DEBUG */

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Irecv( buf, count, datatype, source, tag, comm, request );

#ifdef DEBUG
  printf("Node: %d: Irecv: request = %lx\n", myrank, *request);
#endif /* DEBUG */

  TAU_PROFILE_STOP(tautimer);

  if (TauEnv_get_track_message()) {
    TauAddRequestData(RQ_RECV, count, datatype, source, tag, comm, request, returnVal, 0);
  }

  return returnVal;
}

int  MPI_Irsend( buf, count, datatype, dest, tag, comm, request )
TAU_MPICH3_CONST void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;
  int typesize3 = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Irsend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize3 );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), count * typesize3);
    }
  }
  TAU_PLUGIN_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), count * typesize3, 0);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Irsend( buf, count, datatype, dest, tag, comm, request );


  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Isend( buf, count, datatype, dest, tag, comm, request )
TAU_MPICH3_CONST void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;
  int typesize3 = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Isend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize3 );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), count * typesize3);
    }
  }
  TAU_PLUGIN_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), count * typesize3, 0);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Isend( buf, count, datatype, dest, tag, comm, request );
  TAU_PROFILE_STOP(tautimer);
  return returnVal;
}

int  MPI_Issend( buf, count, datatype, dest, tag, comm, request )
TAU_MPICH3_CONST void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
MPI_Request * request;
{
  int  returnVal;
  int typesize3 = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Issend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize3 );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), count * typesize3);
    }
  }
  TAU_PLUGIN_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), count * typesize3, 0);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Issend( buf, count, datatype, dest, tag, comm, request );
  TAU_PROFILE_STOP(tautimer);
  return returnVal;
}

int   MPI_Pack( inbuf, incount, type, outbuf, outcount, position, comm )
TAU_MPICH3_CONST void * inbuf;
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

  TAU_TRACK_COMM(comm);
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

  TAU_TRACK_COMM(comm);
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

  TAU_TRACK_COMM(comm);
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

  // plugins need the status, too
  //if (TauEnv_get_track_message()) {
    if (status == MPI_STATUS_IGNORE) {
      status = &local_status;
    }
  //}

  TAU_TRACK_COMM(comm);

  TAU_MSG_RECV_PROLOG();
  returnVal = PMPI_Recv( buf, count, datatype, source, tag, comm, status );

  if (source != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    /* note that status->MPI_COMM must == comm */
    if (TauEnv_get_track_message()) {
      PMPI_Get_count( status, MPI_BYTE, &size );
      TAU_TRACE_RECVMSG(status->MPI_TAG, TauTranslateRankToWorld(comm, status->MPI_SOURCE), size);
    }
    int typesize = 0;
    if (datatype != MPI_DATATYPE_NULL) {
      PMPI_Type_size( datatype, &typesize );
    }
    if (status == NULL) {
        TAU_PLUGIN_RECVMSG(tag, TauTranslateRankToWorld(comm, source), count*typesize, 0);
    } else {
        TAU_PLUGIN_RECVMSG(status->MPI_TAG, TauTranslateRankToWorld(comm, status->MPI_SOURCE), count*typesize, 0);
    }
  }
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}


int  MPI_Rsend( buf, count, datatype, dest, tag, comm )
TAU_MPICH3_CONST void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
{
  int  returnVal;
  int typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Rsend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), typesize*count);
    }
  }
  TAU_PLUGIN_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), typesize*count, 0);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Rsend( buf, count, datatype, dest, tag, comm );
  TAU_PROFILE_STOP(tautimer);
  return returnVal;
}

int  MPI_Rsend_init( buf, count, datatype, dest, tag, comm, request )
TAU_MPICH3_CONST void * buf;
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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Rsend_init( buf, count, datatype, dest, tag, comm, request );

if (TauEnv_get_track_message()) {
  TauAddRequestData(RQ_SEND, count, datatype, dest, tag, comm, request, returnVal, 1);
}

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

extern long Tau_get_message_send_path(void);

int  MPI_Send( buf, count, datatype, dest, tag, comm )
TAU_MPICH3_CONST void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
{
  int  returnVal;
  int typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Send()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  TAU_MSG_SEND_PROLOG();

  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), typesize*count);
    }
  }
  TAU_PLUGIN_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), typesize*count, 0);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Send( buf, count, datatype, dest, tag, comm );
  TAU_PROFILE_PARAM1L(Tau_get_message_send_path(), "message send path id");

  TAU_PROFILE_STOP(tautimer);
  return returnVal;
}

int  MPI_Sendrecv( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status )
TAU_MPICH3_CONST void * sendbuf;
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
  MPI_Status local_status;
  int typesize1 = 0;
  int count;


  TAU_PROFILE_TIMER(tautimer, "MPI_Sendrecv()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  if (sendtype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( sendtype, &typesize1 );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(sendtag, TauTranslateRankToWorld(comm, dest), typesize1*sendcount);
    }
  }

    // plugins need the status, too
    if (status == MPI_STATUS_IGNORE) {
      status = &local_status;
    }

  TAU_PLUGIN_SENDMSG(sendtag, TauTranslateRankToWorld(comm, dest), typesize1*sendcount, 0);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Sendrecv( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status );

  if (source != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    if (TauEnv_get_track_message()) {
      PMPI_Get_count( status, MPI_BYTE, &count );
      TAU_TRACE_RECVMSG(status->MPI_TAG, TauTranslateRankToWorld(comm, status->MPI_SOURCE), count);
    }
    int typesize = 0;
    if (recvtype != MPI_DATATYPE_NULL) {
      PMPI_Type_size( recvtype, &typesize );
    }
    if (status == NULL) {
        TAU_PLUGIN_RECVMSG(recvtag, TauTranslateRankToWorld(comm, source), count*typesize, 0);
    } else {
        TAU_PLUGIN_RECVMSG(status->MPI_TAG, TauTranslateRankToWorld(comm, status->MPI_SOURCE), count*typesize, 0);
    }
  }
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
  MPI_Status local_status;
  int size1;
  int typesize2 = 0;


  TAU_PROFILE_TIMER(tautimer, "MPI_Sendrecv_replace()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize2 );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(sendtag, TauTranslateRankToWorld(comm, dest), typesize2*count);
    }
  }

    // plugins need the status, too
    if (status == MPI_STATUS_IGNORE) {
      status = &local_status;
    }
  TAU_PLUGIN_SENDMSG(sendtag, TauTranslateRankToWorld(comm, dest), typesize2*count, 0);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Sendrecv_replace( buf, count, datatype, dest, sendtag, source, recvtag, comm, status );

  if (dest != MPI_PROC_NULL && returnVal == MPI_SUCCESS) {
    if (TauEnv_get_track_message()) {
      PMPI_Get_count( status, MPI_BYTE, &size1 );
      TAU_TRACE_RECVMSG(status->MPI_TAG, TauTranslateRankToWorld(comm, status->MPI_SOURCE), size1);
    }
    int typesize = 0;
    if (datatype != MPI_DATATYPE_NULL) {
      PMPI_Type_size( datatype, &typesize );
    }
    if (status == NULL) {
        TAU_PLUGIN_RECVMSG(recvtag, TauTranslateRankToWorld(comm, source), count*typesize, 0);
    } else {
        TAU_PLUGIN_RECVMSG(status->MPI_TAG, TauTranslateRankToWorld(comm, status->MPI_SOURCE), count*typesize, 0);
    }
  }
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Ssend( buf, count, datatype, dest, tag, comm )
TAU_MPICH3_CONST void * buf;
int count;
MPI_Datatype datatype;
int dest;
int tag;
MPI_Comm comm;
{
  int  returnVal;
  int typesize = 0;

  TAU_PROFILE_TIMER(tautimer, "MPI_Ssend()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  if (datatype != MPI_DATATYPE_NULL) {
    PMPI_Type_size( datatype, &typesize );
  }
  if (TauEnv_get_track_message()) {
    if (dest != MPI_PROC_NULL) {
      TAU_TRACE_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), typesize*count);
    }
  }
  TAU_PLUGIN_SENDMSG(tag, TauTranslateRankToWorld(comm, dest), typesize*count, 0);

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Ssend( buf, count, datatype, dest, tag, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Ssend_init( buf, count, datatype, dest, tag, comm, request )
TAU_MPICH3_CONST void * buf;
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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Ssend_init( buf, count, datatype, dest, tag, comm, request );

if (TauEnv_get_track_message()) {
  TauAddRequestData(RQ_SEND, count, datatype, dest, tag, comm, request, returnVal, 1);
}

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Start( request )
MPI_Request * request;
{
  request_data * rq;
  int  returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Start()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (TauEnv_get_track_message()) {
    rq = TauGetRequestData(request);
    TauProcessSend(request, "MPI_Start");
  }

  returnVal = PMPI_Start( request );

  if (TauEnv_get_track_message()) {
    /* fix up the request since MPI_Start may (will) change it */
    rq->request = request;
  }

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
  MPI_Request saverequest;
  MPI_Status local_status;

  TAU_PROFILE_TIMER(tautimer, "MPI_Test()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (TauEnv_get_track_message()) {
    saverequest = *request;
    if (status == MPI_STATUS_IGNORE) {
      status = &local_status;
    }
  }

  returnVal = PMPI_Test( request, flag, status );

  if (TauEnv_get_track_message()) {
    if (*flag) {
      TauProcessRecv(&saverequest, status, "MPI_Test");
    }
  }

  TAU_PROFILE_STOP(tautimer);
  return returnVal;
}

int  MPI_Testall( count, array_of_requests, flag, array_of_statuses )
int count;
MPI_Request * array_of_requests;
int * flag;
MPI_Status * array_of_statuses;
{
  int returnVal;
  int need_to_free = 0;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];


  TAU_PROFILE_TIMER(tautimer, "MPI_Testall()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  if (TauEnv_get_track_message()) {
    for (i = 0; i < count; i++) {
      saverequest[i] = array_of_requests[i];
    }
    if (array_of_statuses == MPI_STATUSES_IGNORE) {
      array_of_statuses = (MPI_Status*) malloc (sizeof(MPI_Status)*count);
      need_to_free = 1;
    }
  }

  returnVal = PMPI_Testall( count, array_of_requests, flag, array_of_statuses );

  if (TauEnv_get_track_message()) {
    if (*flag) {
      /* at least one completed */
      for(i=0; i < count; i++) {
	TauProcessRecv(&saverequest[i], &array_of_statuses[i], "MPI_Testall");
      }
    }
    if (need_to_free) {
      free(array_of_statuses);
    }
  }

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
  MPI_Status local_status;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];

  TAU_PROFILE_TIMER(tautimer, "MPI_Testany()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (TauEnv_get_track_message()) {
    for (i = 0; i < count; i++) {
      saverequest[i] = array_of_requests[i];
    }
    if (status == MPI_STATUS_IGNORE) {
      status = &local_status;
    }
  }

  returnVal = PMPI_Testany( count, array_of_requests, index, flag, status );


  if (TauEnv_get_track_message()) {
    if (*flag && (*index != MPI_UNDEFINED)) {
      TauProcessRecv(&saverequest[*index], status, "MPI_Testany");
    }

  }
  TAU_PROFILE_STOP(tautimer);
  return returnVal;
}

int  MPI_Test_cancelled( status, flag )
TAU_MPICH3_CONST MPI_Status * status;
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
  int need_to_free = 0;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];

  TAU_PROFILE_TIMER(tautimer, "MPI_Testsome()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (TauEnv_get_track_message()) {
    for (i = 0; i < incount; i++){
      saverequest[i] = array_of_requests[i];
    }
    if (array_of_statuses == MPI_STATUSES_IGNORE) {
      array_of_statuses = (MPI_Status*) malloc (sizeof(MPI_Status)*incount);
      need_to_free = 1;
    }
  }

  returnVal = PMPI_Testsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses );

  if (TauEnv_get_track_message()) {
    for (i=0; i < *outcount; i++) {
      TauProcessRecv( &saverequest[array_of_indices[i]],
		      &(array_of_statuses[i]),
		      "MPI_Testsome" );
    }
    if (need_to_free) {
    free(array_of_statuses);
    }
  }

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

#if (defined(TAU_SGI_MPT_MPI) || defined(TAU_MPI_HINDEX_CONST)) || MPI_VERSION > 2
#define TAU_HINDEXED_CONST const
#else
#ifndef TAU_HINDEXED_CONST
#define TAU_HINDEXED_CONST
#endif /* TAU_HINDEXED_CONST */
#endif /* TAU_SGI_MPT_MPI */

// OpenMPI 4 and later have removed some functions deleted in MPI 3.0
//#if !defined(OMPI_MAJOR_VERSION) || (OMPI_MAJOR_VERSION < 4)
#if MPI_VERSION < 2

int  MPI_Type_hindexed( count, blocklens, indices, old_type, newtype )
int count;
TAU_HINDEXED_CONST int * blocklens;
TAU_HINDEXED_CONST MPI_Aint * indices;
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

#else /* MPI_VERSION < 2 */

/* replacements defined in TauMpiExtensions.c, for some reason */

#endif /* MPI_VERSION < 2 */

int  MPI_Type_indexed( count, blocklens, indices, old_type, newtype )
int count;
TAU_MPICH3_CONST int * blocklens;
TAU_MPICH3_CONST int * indices;
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

// OpenMPI 4 and later have removed some functions deleted in MPI 3.0
// #if !defined(OMPI_MAJOR_VERSION) || (OMPI_MAJOR_VERSION < 4)
#if MPI_VERSION < 2

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

#else /* MPI_VERSION < 2 */
#endif /* MPI_VERSION < 2 */

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


// OpenMPI 4 and later have removed some functions deleted in MPI 3.0
// #if !defined(OMPI_MAJOR_VERSION) || (OMPI_MAJOR_VERSION < 4)
#if MPI_VERSION < 2

int  MPI_Type_struct( count, blocklens, indices, old_types, newtype )
int count;
TAU_OPENMPI3_CONST int * blocklens;
TAU_OPENMPI3_CONST MPI_Aint * indices;
TAU_OPENMPI3_CONST MPI_Datatype * old_types;
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

#else /* MPI_VERSION < 2 */

/* replacements defined in TauMpiExtensions.c, for some reason */

#endif /* MPI_VERSION < 2 */

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
TAU_MPICH3_CONST void * inbuf;
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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Unpack( inbuf, insize, position, outbuf, outcount, type, comm );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Wait( request, status )
MPI_Request * request;
MPI_Status * status;
{
  int   returnVal;
  MPI_Status local_status;
  MPI_Request saverequest;

  TAU_PROFILE_TIMER(tautimer, "MPI_Wait()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (TauEnv_get_track_message()) {
    if (status == MPI_STATUS_IGNORE) {
      status = &local_status;
    }
    saverequest = *request;
  }

  returnVal = PMPI_Wait( request, status );

  if (TauEnv_get_track_message()) {
    TauProcessRecv(&saverequest, status, "MPI_Wait");
  }

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int  MPI_Waitall( count, array_of_requests, array_of_statuses )
int count;
MPI_Request * array_of_requests;
MPI_Status * array_of_statuses;
{
  int  returnVal;
  int need_to_free = 0;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];

  TAU_PROFILE_TIMER(tautimer, "MPI_Waitall()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (TauEnv_get_track_message()) {
    for (i = 0; i < count; i++) {
      saverequest[i] = array_of_requests[i];
    }

    if (array_of_statuses == MPI_STATUSES_IGNORE) {
      array_of_statuses = (MPI_Status*) malloc (sizeof(MPI_Status)*count);
      need_to_free = 1;
    }
  }

  returnVal = PMPI_Waitall( count, array_of_requests, array_of_statuses );

  if (TauEnv_get_track_message()) {
    for(i=0; i < count; i++) {
      TauProcessRecv(&saverequest[i], &array_of_statuses[i], "MPI_Waitall");
    }

    if (need_to_free) {
      free(array_of_statuses);
    }
  }

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
  MPI_Status local_status;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];

  TAU_PROFILE_TIMER(tautimer, "MPI_Waitany()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (TauEnv_get_track_message()) {
    for (i = 0; i < count; i++){
      saverequest[i] = array_of_requests[i];
    }
    if (status == MPI_STATUS_IGNORE) {
      status = &local_status;
    }
  }

  returnVal = PMPI_Waitany( count, array_of_requests, index, status );


  if (TauEnv_get_track_message()) {
    TauProcessRecv( &saverequest[*index], status, "MPI_Waitany" );
  }

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

  int need_to_free = 0;
  int i;
  MPI_Request saverequest[TAU_MAX_REQUESTS];

  TAU_PROFILE_TIMER(tautimer, "MPI_Waitsome()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  if (TauEnv_get_track_message()) {
    for (i = 0; i < incount; i++) {
      saverequest[i] = array_of_requests[i];
    }

    if (array_of_statuses == MPI_STATUSES_IGNORE) {
      array_of_statuses = (MPI_Status*) malloc (sizeof(MPI_Status)*incount);
      need_to_free = 1;
    }
  }

  returnVal = PMPI_Waitsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses );


  if (TauEnv_get_track_message()) {
    for (i=0; i < *outcount; i++) {
      TauProcessRecv( &saverequest[array_of_indices[i]], &(array_of_statuses[i]), "MPI_Waitsome" );
    }
    if (need_to_free) {
      free(array_of_statuses);
    }

  }
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

#ifndef TAU_MPI_DISABLE_COMM_WRAPPERS
int   MPI_Cart_coords( comm, rank, maxdims, coords )
MPI_Comm comm;
int rank;
int maxdims;
int * coords;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_coords()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_TRACK_COMM(comm);

  returnVal = PMPI_Cart_coords( comm, rank, maxdims, coords );

  //TIMER_EXIT_CART_COORDS_EVENT( comm, rank, maxdims, coords );
  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_create( comm_old, ndims, dims, periods, reorder, comm_cart )
MPI_Comm comm_old;
int ndims;
TAU_MPICH3_CONST int * dims;
TAU_MPICH3_CONST int * periods;
int reorder;
MPI_Comm * comm_cart;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_create()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  returnVal = PMPI_Cart_create( comm_old, ndims, dims, periods, reorder, comm_cart );

  TIMER_EXIT_CART_CREATE_EVENT(comm_old, ndims, dims, periods, reorder, *comm_cart);
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
TAU_MPICH3_CONST int * dims;
TAU_MPICH3_CONST int * periods;
int * newrank;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_map()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  TAU_TRACK_COMM(comm_old);
  returnVal = PMPI_Cart_map( comm_old, ndims, dims, periods, newrank );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Cart_rank( comm, coords, rank )
MPI_Comm comm;
TAU_MPICH3_CONST int * coords;
int * rank;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_rank()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  TAU_TRACK_COMM(comm);
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
TAU_MPICH3_CONST int * remain_dims;
MPI_Comm * comm_new;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Cart_sub()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);
  TAU_TRACK_COMM(comm);

  returnVal = PMPI_Cart_sub( comm, remain_dims, comm_new );

  TIMER_EXIT_CART_SUB_EVENT(comm,remain_dims,*comm_new);
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

  TAU_TRACK_COMM(comm);
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
TAU_MPICH3_CONST int * index;
TAU_MPICH3_CONST int * edges;
int reorder;
MPI_Comm * comm_graph;
{
  int   returnVal;

  TAU_PROFILE_TIMER(tautimer, "MPI_Graph_create()",  " ", TAU_MESSAGE);
  TAU_PROFILE_START(tautimer);

  TAU_TRACK_COMM(comm_old);
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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Graph_get( comm, maxindex, maxedges, index, edges );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}

int   MPI_Graph_map( comm_old, nnodes, index, edges, newrank )
MPI_Comm comm_old;
int nnodes;
TAU_MPICH3_CONST int * index;
TAU_MPICH3_CONST int * edges;
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

  TAU_TRACK_COMM(comm);
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

  TAU_TRACK_COMM(comm);
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

  TAU_TRACK_COMM(comm);
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

  TAU_TRACK_COMM(comm);
  returnVal = PMPI_Topo_test( comm, top_type );

  TAU_PROFILE_STOP(tautimer);

  return returnVal;
}
#endif // TAU_MPI_DISABLE_COMM_WRAPPERS


//For a given process, process is the unique MPI rank
//Node n is the nth node in the allocation
//Core m is the mth core on node n
int TauGetCpuSite(int *node, int *core, int *rank) {
  char host_name[MPI_MAX_PROCESSOR_NAME];
  char (*host_names)[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm internode;
  MPI_Comm intranode;

  int nprocs, namelen,n,bytes;

  PMPI_Comm_rank(MPI_COMM_WORLD, rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  PMPI_Get_processor_name(host_name,&namelen);
  bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);

  host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);

  strcpy(host_names[*rank], host_name);
  for (n=0; n<nprocs; n++) {
    PMPI_Bcast(host_names[n],MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
  }

  unsigned int color;
  color = 0;

  for (n=1; n<nprocs; n++) {
    if(strcmp(host_names[n-1], host_names[n])) color++;
    if(strcmp(host_name, host_names[n]) == 0) break;
  }

  PMPI_Comm_split(MPI_COMM_WORLD, color, *rank, &internode);
  PMPI_Comm_rank(internode,core);

  PMPI_Comm_split(MPI_COMM_WORLD, *core, *rank, &intranode);

  PMPI_Comm_rank(intranode,node);
  return 0;
}

/* moved over to TauUnify.o so ScoreP can use it with tau_run */
/*
int TauGetMpiRank(void)
{
  int rank;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}
*/


char * Tau_printRanks(void *comm_ptr) {
  /* Create an array of ranks and fill it in using MPI_Group_translate_ranks*/
  /* Fill in a character array that we can append to the name and make it accessible using a map */

  int i, limit, size;
  char name[16384];
  char rankbuffer[256];
  int worldrank;
  MPI_Comm comm = (MPI_Comm)(intptr_t) comm_ptr;
  memset(name, 0, 16384);

  PMPI_Comm_size(comm, &size);
  limit = (size < TAU_MAX_MPI_RANKS) ? size : TAU_MAX_MPI_RANKS;
  for ( i = 0; i < limit; i++) {
    worldrank = TauTranslateRankToWorld(comm, i);
    if (i == 0) {
      sprintf(rankbuffer, "ranks: %d", worldrank);
    } else {
      sprintf(rankbuffer, ", %d", worldrank);
    }
    strcat(name, rankbuffer);
  }
  if (limit < size) {
    strcat(name, " ...");
  }
  sprintf(rankbuffer,"> <addr=%p", comm_ptr);
  strcat(name, rankbuffer);
  return strdup(name);


}

/* EOF TauMpi.c */
