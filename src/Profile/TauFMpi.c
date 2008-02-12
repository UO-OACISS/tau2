/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/tau             **
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
**      Documentation   : See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#include <mpi.h>
#include <Profile/TauUtil.h>


#ifdef TAU_LAMPI
MPI_Fint TAU_MPI_Request_c2f(MPI_Request c_request) {
  MPI_Fint f_request;
  f_request = MPI_Request_c2f(c_request);
  /* LA-MPI doesn't seem to translate MPI_REQUEST_NULL properly
     so we'll check for it and set it to the proper value for fortran */
  if (c_request == MPI_REQUEST_NULL) {
    f_request = -1;
  }
  return f_request;
}
#else
/* For all other implementations, just #define it to avoid a wrapper function call */
#define TAU_MPI_Request_c2f MPI_Request_c2f
#endif /* TAU_LAMPI */



/******************************************************/
/******************************************************/
void  mpi_allgather_( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Allgather( sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, *recvcount, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm) );

}

void  mpi_allgather__( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allgather_( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr);
}

void  MPI_ALLGATHER( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allgather_( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr);
}

void  MPI_ALLGATHER_( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allgather_( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr);
}


void  mpi_allgather( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allgather_( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm , ierr);
}


/******************************************************/
/******************************************************/

void  mpi_allgatherv_( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcounts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Allgatherv( sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, recvcounts, displs, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm) );

}

void  mpi_allgatherv__( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcounts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allgatherv_( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr);
}

void  MPI_ALLGATHERV( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcounts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allgatherv_( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr);
}

void  MPI_ALLGATHERV_( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcounts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allgatherv_( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr);
}

void  mpi_allgatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcounts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allgatherv_( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm , ierr);
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPICH2_MPI_IN_PLACE
extern int MPIR_F_MPI_IN_PLACE; 
#endif /* TAU_MPICH2_MPI_IN_PLACE */

void   mpi_allreduce_( sendbuf, recvbuf, count, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
#ifdef TAU_MPICH2_MPI_IN_PLACE
    if (sendbuf == (void *) MPIR_F_MPI_IN_PLACE)
    {
      *ierr = MPI_Allreduce( MPI_IN_PLACE, recvbuf, *count, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), MPI_Comm_f2c(*comm) );

    }
    else
#endif /* TAU_MPICH2_MPI_IN_PLACE */
      *ierr = MPI_Allreduce( sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), MPI_Comm_f2c(*comm) );

}

void   mpi_allreduce__( sendbuf, recvbuf, count, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allreduce_( sendbuf, recvbuf, count, datatype, op, comm , ierr);
}

void   MPI_ALLREDUCE( sendbuf, recvbuf, count, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allreduce_( sendbuf, recvbuf, count, datatype, op, comm , ierr);
}
 
void   MPI_ALLREDUCE_( sendbuf, recvbuf, count, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_allreduce_( sendbuf, recvbuf, count, datatype, op, comm , ierr);
}

#ifdef TAU_IBM_MPI
extern int mpi_in_place_;
#endif /* TAU_IBM_MPI */

void   mpi_allreduce( sendbuf, recvbuf, count, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
#ifdef TAU_IBM_MPI
  
  if ((int *)sendbuf == &mpi_in_place_)
  { /* FOR IBM ! */
#ifdef DEBUG_PROF
    printf("mpi_in_place_ = %d, s = %llx, &m = %llx \n", 
	mpi_in_place_, sendbuf, &mpi_in_place_);
#endif /* DEBUG_PROF */
    mpi_allreduce_( MPI_IN_PLACE, recvbuf, count, datatype, op, comm , ierr);
  }
  else
#endif /* TAU_IBM_MPI */
    mpi_allreduce_( sendbuf, recvbuf, count, datatype, op, comm , ierr);

}

/******************************************************/
/******************************************************/

void  mpi_alltoall_( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr; 
{
  *ierr = MPI_Alltoall(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, *recvcnt, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm) );
}

void  mpi_alltoall__( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr; 
{
  mpi_alltoall_( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr);
}

void  MPI_ALLTOALL( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr; 
{
  mpi_alltoall_( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr);
}

void  MPI_ALLTOALL_( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_alltoall_( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm
, ierr);
}

void  mpi_alltoall( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_alltoall_( sendbuf, sendcount, sendtype, recvbuf, recvcnt, recvtype, comm , ierr);
}

/******************************************************/
/******************************************************/

void   mpi_alltoallv_( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * sdispls;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * rdispls;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Alltoallv( sendbuf, sendcnts, sdispls, MPI_Type_f2c(*sendtype), recvbuf, recvcnts, rdispls, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm) );

}

void   mpi_alltoallv__( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * sdispls;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * rdispls;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_alltoallv_( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr);
}

void   MPI_ALLTOALLV( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * sdispls;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * rdispls;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_alltoallv_( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr);
}

void   MPI_ALLTOALLV_( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * sdispls;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * rdispls;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_alltoallv_( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr);
}


void   mpi_alltoallv( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * sdispls;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * rdispls;
MPI_Fint *recvtype;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_alltoallv_( sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm , ierr);
}


/******************************************************/
/******************************************************/
void   mpi_barrier_( comm , ierr)
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Barrier( MPI_Comm_f2c(*comm) );
}

void   mpi_barrier__( comm , ierr)
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_barrier_( comm , ierr);
}

void   MPI_BARRIER( comm , ierr)
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_barrier_( comm , ierr);
}

void   MPI_BARRIER_( comm , ierr)
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_barrier_( comm , ierr);
}

void   mpi_barrier( comm , ierr)
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_barrier_( comm , ierr);
}

/******************************************************/
/******************************************************/
void   mpi_bcast_( buffer, count, datatype, root, comm , ierr)
void * buffer;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Bcast( buffer, *count, MPI_Type_f2c(*datatype), *root, MPI_Comm_f2c(*comm) );
}

void   mpi_bcast__( buffer, count, datatype, root, comm , ierr)
void * buffer;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_bcast_( buffer, count, datatype, root, comm , ierr);
}

void   MPI_BCAST( buffer, count, datatype, root, comm , ierr)
void * buffer;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_bcast_( buffer, count, datatype, root, comm , ierr);
}

void   MPI_BCAST_( buffer, count, datatype, root, comm , ierr)
void * buffer;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_bcast_( buffer, count, datatype, root, comm , ierr);
}

void   mpi_bcast( buffer, count, datatype, root, comm , ierr)
void * buffer;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_bcast_( buffer, count, datatype, root, comm , ierr);
}


/******************************************************/
/******************************************************/

void  mpi_gather_( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  
  *ierr = MPI_Gather( sendbuf, *sendcnt, MPI_Type_f2c(*sendtype), recvbuf, *recvcount, MPI_Type_f2c(*recvtype), *root, MPI_Comm_f2c(*comm) );

}


void  mpi_gather__( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_gather_( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr);
}

void  MPI_GATHER( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_gather_( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr);
}

void  MPI_GATHER_( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_gather_( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr);
}


void  mpi_gather( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_gather_( sendbuf, sendcnt, sendtype, recvbuf, recvcount, recvtype, root, comm , ierr);
}


/******************************************************/
/******************************************************/

void mpi_gatherv_( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Gatherv( sendbuf, *sendcnt, MPI_Type_f2c(*sendtype), recvbuf, recvcnts, displs, MPI_Type_f2c(*recvtype), *root, MPI_Comm_f2c(*comm) );

}

void mpi_gatherv__( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_gatherv_( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr); 
}

void MPI_GATHERV( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_gatherv_( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr); 
}

void MPI_GATHERV_( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_gatherv_( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr);
}

void mpi_gatherv( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint * displs;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_gatherv_( sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm , ierr); 
}

/******************************************************/
/******************************************************/

void mpi_op_create_( function, commute, op , ierr)
MPI_User_function * function;
MPI_Fint *commute;
MPI_Fint * op;
MPI_Fint *ierr;
{
  MPI_Op local_op; 
  *ierr = MPI_Op_create( function, *commute, &local_op );
  *op = MPI_Op_c2f(local_op);

}

void mpi_op_create__( function, commute, op , ierr)
MPI_User_function * function;
MPI_Fint *commute;
MPI_Fint * op;
MPI_Fint *ierr;
{
  mpi_op_create_( function, commute, op , ierr);
}

void MPI_OP_CREATE( function, commute, op , ierr)
MPI_User_function * function;
MPI_Fint *commute;
MPI_Fint * op;
MPI_Fint *ierr;
{
  mpi_op_create_( function, commute, op , ierr);
}

void MPI_OP_CREATE_( function, commute, op , ierr)
MPI_User_function * function;
MPI_Fint *commute;
MPI_Fint * op;
MPI_Fint *ierr;
{
  mpi_op_create_( function, commute, op , ierr);
}

void mpi_op_create( function, commute, op , ierr)
MPI_User_function * function;
MPI_Fint *commute;
MPI_Fint * op;
MPI_Fint *ierr;
{
  mpi_op_create_( function, commute, op , ierr);
}


/******************************************************/
/******************************************************/

void  mpi_op_free_( op , ierr)
MPI_Fint * op;
MPI_Fint *ierr;
{
  MPI_Op local_op = MPI_Op_f2c(*op);
  *ierr = MPI_Op_free( &local_op );
  *op = MPI_Op_c2f(local_op);
}


void  mpi_op_free__( op , ierr)
MPI_Fint * op;
MPI_Fint *ierr;
{
  mpi_op_free_( op , ierr); 
}

void  MPI_OP_FREE( op , ierr)
MPI_Fint * op;
MPI_Fint *ierr;
{
  mpi_op_free_( op , ierr); 
} 

void  MPI_OP_FREE_( op , ierr)
MPI_Fint * op;
MPI_Fint *ierr;
{
  mpi_op_free_( op , ierr);
}

void  mpi_op_free( op , ierr)
MPI_Fint * op;
MPI_Fint *ierr;
{
  mpi_op_free_( op , ierr); 
} 


/******************************************************/
/******************************************************/

void mpi_reduce_scatter_( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Reduce_scatter( sendbuf, recvbuf, recvcnts, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), MPI_Comm_f2c(*comm) );
}

void mpi_reduce_scatter__( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_reduce_scatter_( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr);
}

void MPI_REDUCE_SCATTER( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_reduce_scatter_( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr);
}

void MPI_REDUCE_SCATTER_( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_reduce_scatter_( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr);
}

void mpi_reduce_scatter( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint * recvcnts;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_reduce_scatter_( sendbuf, recvbuf, recvcnts, datatype, op, comm , ierr);
}

/******************************************************/
/******************************************************/

void mpi_reduce_( sendbuf, recvbuf, count, datatype, op, root, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Reduce( sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), *root, MPI_Comm_f2c(*comm) );
}

void mpi_reduce__( sendbuf, recvbuf, count, datatype, op, root, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_reduce_( sendbuf, recvbuf, count, datatype, op, root, comm , ierr);
}

void MPI_REDUCE( sendbuf, recvbuf, count, datatype, op, root, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_reduce_( sendbuf, recvbuf, count, datatype, op, root, comm , ierr);
}

void MPI_REDUCE_( sendbuf, recvbuf, count, datatype, op, root, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_reduce_( sendbuf, recvbuf, count, datatype, op, root, comm , ierr);
}


void mpi_reduce( sendbuf, recvbuf, count, datatype, op, root, comm , ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_reduce_( sendbuf, recvbuf, count, datatype, op, root, comm , ierr);
}

/******************************************************/
/******************************************************/

void mpi_scan_( sendbuf, recvbuf, count, datatype, op, comm, ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Scan( sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), MPI_Comm_f2c(*comm) );
}

void mpi_scan__( sendbuf, recvbuf, count, datatype, op, comm, ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scan_( sendbuf, recvbuf, count, datatype, op, comm, ierr);
}

void MPI_SCAN( sendbuf, recvbuf, count, datatype, op, comm, ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scan_( sendbuf, recvbuf, count, datatype, op, comm, ierr);
}

void MPI_SCAN_( sendbuf, recvbuf, count, datatype, op, comm, ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scan_( sendbuf, recvbuf, count, datatype, op, comm, ierr);
}

void mpi_scan( sendbuf, recvbuf, count, datatype, op, comm, ierr)
void * sendbuf;
void * recvbuf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *op;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scan_( sendbuf, recvbuf, count, datatype, op, comm, ierr);
}

/******************************************************/
/******************************************************/

void   mpi_scatter_( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr )
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Scatter( sendbuf, *sendcnt, MPI_Type_f2c(*sendtype), recvbuf, *recvcnt, MPI_Type_f2c(*recvtype), *root, MPI_Comm_f2c(*comm) );
}

void   mpi_scatter__( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr )
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scatter_( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr );
}

void   MPI_SCATTER( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr )
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scatter_( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr );
}

void   MPI_SCATTER_( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr )
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scatter_( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr );
}

void   mpi_scatter( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr )
void * sendbuf;
MPI_Fint *sendcnt;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scatter_( sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm, ierr );
}


/******************************************************/
/******************************************************/

void  mpi_scatterv_( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * displs;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Scatterv( sendbuf, sendcnts, displs, MPI_Type_f2c(*sendtype), recvbuf, *recvcnt, MPI_Type_f2c(*recvtype), *root, MPI_Comm_f2c(*comm) );
}

void  mpi_scatterv__( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * displs;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{ 
  mpi_scatterv_( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr);
}

void  MPI_SCATTERV( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * displs;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scatterv_( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr);
}

void  MPI_SCATTERV_( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * displs;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scatterv_( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr);
}

void  mpi_scatterv( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr)
void * sendbuf;
MPI_Fint * sendcnts;
MPI_Fint * displs;
MPI_Fint *sendtype;
void * recvbuf;
MPI_Fint *recvcnt;
MPI_Fint *recvtype;
MPI_Fint *root;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_scatterv_( sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm , ierr);
}


/******************************************************/
/******************************************************/

void   mpi_attr_delete_( comm, keyval, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
MPI_Fint *ierr;
{
  *ierr = MPI_Attr_delete( MPI_Comm_f2c(*comm), *keyval );
}

void   mpi_attr_delete__( comm, keyval, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
MPI_Fint *ierr;
{
  mpi_attr_delete_( comm, keyval, ierr);
}

void   MPI_ATTR_DELETE( comm, keyval, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
MPI_Fint *ierr;
{
  mpi_attr_delete_( comm, keyval, ierr);
}

void   MPI_ATTR_DELETE_( comm, keyval, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
MPI_Fint *ierr;
{
  mpi_attr_delete_( comm, keyval, ierr);
}

void   mpi_attr_delete( comm, keyval, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
MPI_Fint *ierr;
{
  mpi_attr_delete_( comm, keyval, ierr);
}


/******************************************************/
/******************************************************/

void mpi_attr_get_( comm, keyval, attr_value, flag , ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  *ierr = MPI_Attr_get( MPI_Comm_f2c(*comm), *keyval, attr_value, flag );
}

void mpi_attr_get__( comm, keyval, attr_value, flag , ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_attr_get_( comm, keyval, attr_value, flag , ierr);
}

void MPI_ATTR_GET( comm, keyval, attr_value, flag , ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_attr_get_( comm, keyval, attr_value, flag , ierr);
}

void MPI_ATTR_GET_( comm, keyval, attr_value, flag , ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_attr_get_( comm, keyval, attr_value, flag , ierr);
}

void mpi_attr_get( comm, keyval, attr_value, flag , ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_attr_get_( comm, keyval, attr_value, flag , ierr);
}

/******************************************************/
/******************************************************/

void   mpi_attr_put_( comm, keyval, attr_value, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint *ierr;
{
  *ierr = MPI_Attr_put( MPI_Comm_f2c(*comm), *keyval, attr_value );
}

void   mpi_attr_put__( comm, keyval, attr_value, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint *ierr;
{
  mpi_attr_put_( comm, keyval, attr_value, ierr);
}

void   MPI_ATTR_PUT( comm, keyval, attr_value, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint *ierr;
{
  mpi_attr_put_( comm, keyval, attr_value, ierr);
}

void   MPI_ATTR_PUT_( comm, keyval, attr_value, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint *ierr;
{
  mpi_attr_put_( comm, keyval, attr_value, ierr);
}

void   mpi_attr_put( comm, keyval, attr_value, ierr)
MPI_Fint *comm;
MPI_Fint *keyval;
void * attr_value;
MPI_Fint *ierr;
{
  mpi_attr_put_( comm, keyval, attr_value, ierr);
}

/******************************************************/
/******************************************************/

void  mpi_comm_compare_( comm1, comm2, result, ierr )
MPI_Fint *comm1;
MPI_Fint *comm2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  *ierr = MPI_Comm_compare( MPI_Comm_f2c(*comm1), MPI_Comm_f2c(*comm2), result );
}

void  mpi_comm_compare__( comm1, comm2, result, ierr )
MPI_Fint *comm1;
MPI_Fint *comm2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  mpi_comm_compare_( comm1, comm2, result, ierr );
}

void  MPI_COMM_COMPARE( comm1, comm2, result, ierr )
MPI_Fint *comm1;
MPI_Fint *comm2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  mpi_comm_compare_( comm1, comm2, result, ierr );
}

void  MPI_COMM_COMPARE_( comm1, comm2, result, ierr )
MPI_Fint *comm1;
MPI_Fint *comm2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  mpi_comm_compare_( comm1, comm2, result, ierr );
}

void  mpi_comm_compare( comm1, comm2, result, ierr )
MPI_Fint *comm1;
MPI_Fint *comm2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  mpi_comm_compare_( comm1, comm2, result, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_comm_create_( comm, group, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *group;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  MPI_Comm local_comm_out;
  *ierr = MPI_Comm_create( MPI_Comm_f2c(*comm), MPI_Group_f2c(*group), &local_comm_out);
  *comm_out = MPI_Comm_c2f(local_comm_out);
}

void  mpi_comm_create__( comm, group, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *group;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_create_( comm, group, comm_out, ierr ); 
}

void  MPI_COMM_CREATE( comm, group, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *group;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_create_( comm, group, comm_out, ierr );
}

void  MPI_COMM_CREATE_( comm, group, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *group;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_create_( comm, group, comm_out, ierr );
}

void  mpi_comm_create( comm, group, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *group;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_create_( comm, group, comm_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_comm_dup_( comm, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  MPI_Comm local_comm_out;
  *ierr = MPI_Comm_dup( MPI_Comm_f2c(*comm), &local_comm_out);
  *comm_out = MPI_Comm_c2f(local_comm_out);
}

void   mpi_comm_dup__( comm, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_dup_( comm, comm_out, ierr );
}

void   MPI_COMM_DUP( comm, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_dup_( comm, comm_out, ierr );
}

void   MPI_COMM_DUP_( comm, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_dup_( comm, comm_out, ierr );
}


void   mpi_comm_dup( comm, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_dup_( comm, comm_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_comm_free_( comm, ierr)
MPI_Fint * comm;
MPI_Fint *ierr;
{
  MPI_Comm local_comm = MPI_Comm_f2c(*comm);
  *ierr = MPI_Comm_free( &local_comm );
  *comm = MPI_Comm_c2f(local_comm);
}

void   mpi_comm_free__( comm, ierr)
MPI_Fint * comm;
MPI_Fint *ierr;
{
  mpi_comm_free_( comm, ierr);
}

void   MPI_COMM_FREE( comm, ierr)
MPI_Fint * comm;
MPI_Fint *ierr;
{
  mpi_comm_free_( comm, ierr);
}

void   MPI_COMM_FREE_( comm, ierr)
MPI_Fint * comm;
MPI_Fint *ierr;
{
  mpi_comm_free_( comm, ierr);
}

void   mpi_comm_free( comm, ierr)
MPI_Fint * comm;
MPI_Fint *ierr;
{
  mpi_comm_free_( comm, ierr);
}


/******************************************************/
/******************************************************/

void   mpi_comm_group_( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Comm_group( MPI_Comm_f2c(*comm), &local_group );
  *group = MPI_Group_c2f(local_group);
}

void   mpi_comm_group__( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_comm_group_( comm, group, ierr );
}

void   MPI_COMM_GROUP( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_comm_group_( comm, group, ierr );
}

void   MPI_COMM_GROUP_( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_comm_group_( comm, group, ierr );
}

void   mpi_comm_group( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_comm_group_( comm, group, ierr );
}


/******************************************************/
/******************************************************/

void   mpi_comm_rank_( comm, rank, ierr )
MPI_Fint *comm;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  *ierr = MPI_Comm_rank( MPI_Comm_f2c(*comm), rank );
}

void   mpi_comm_rank__( comm, rank, ierr )
MPI_Fint *comm;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_comm_rank_( comm, rank, ierr );
}

void   MPI_COMM_RANK( comm, rank, ierr )
MPI_Fint *comm;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_comm_rank_( comm, rank, ierr );
}

void   MPI_COMM_RANK_( comm, rank, ierr )
MPI_Fint *comm;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_comm_rank_( comm, rank, ierr );
}

void   mpi_comm_rank( comm, rank, ierr )
MPI_Fint *comm;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_comm_rank_( comm, rank, ierr );
}


/******************************************************/
/******************************************************/

void   mpi_comm_remote_group_( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Comm_remote_group( MPI_Comm_f2c(*comm), &local_group );
  *group = MPI_Group_c2f(local_group);
}

void   mpi_comm_remote_group__( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_comm_remote_group_( comm, group, ierr );
}

void   MPI_COMM_REMOTE_GROUP( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_comm_remote_group_( comm, group, ierr );
}

void   MPI_COMM_REMOTE_GROUP_( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_comm_remote_group_( comm, group, ierr );
}

void   mpi_comm_remote_group( comm, group, ierr )
MPI_Fint *comm;
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_comm_remote_group_( comm, group, ierr );
}


/******************************************************/
/******************************************************/

void   mpi_comm_remote_size_( comm, size, ierr )
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  *ierr = MPI_Comm_remote_size( MPI_Comm_f2c(*comm), size );
}

void   mpi_comm_remote_size__( comm, size, ierr )
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_comm_remote_size_( comm, size, ierr );
}

void   MPI_COMM_REMOTE_SIZE( comm, size, ierr )
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_comm_remote_size_( comm, size, ierr );
}

void   MPI_COMM_REMOTE_SIZE_( comm, size, ierr )
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_comm_remote_size_( comm, size, ierr );
}

void   mpi_comm_remote_size( comm, size, ierr )
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_comm_remote_size_( comm, size, ierr );
}


/******************************************************/
/******************************************************/

void   mpi_comm_size_( comm, size , ierr)
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  *ierr = MPI_Comm_size( MPI_Comm_f2c(*comm), size );
}

void   mpi_comm_size__( comm, size , ierr)
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_comm_size_( comm, size , ierr);
}

void   MPI_COMM_SIZE( comm, size , ierr)
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_comm_size_( comm, size , ierr);
}

void   MPI_COMM_SIZE_( comm, size , ierr)
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_comm_size_( comm, size , ierr);
}

void   mpi_comm_size( comm, size , ierr)
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_comm_size_( comm, size , ierr);
}


/******************************************************/
/******************************************************/
void   mpi_comm_split_( comm, color, key, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *color;
MPI_Fint *key;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  MPI_Comm local_comm_out;
  *ierr = MPI_Comm_split( MPI_Comm_f2c(*comm), *color, *key, &local_comm_out );
  *comm_out = MPI_Comm_c2f(local_comm_out); 
}

void   mpi_comm_split__( comm, color, key, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *color;
MPI_Fint *key;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_split_( comm, color, key, comm_out, ierr );
}

void   MPI_COMM_SPLIT( comm, color, key, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *color;
MPI_Fint *key;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_split_( comm, color, key, comm_out, ierr );
}

void   MPI_COMM_SPLIT_( comm, color, key, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *color;
MPI_Fint *key;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_split_( comm, color, key, comm_out, ierr );
}


void   mpi_comm_split( comm, color, key, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *color;
MPI_Fint *key;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_comm_split_( comm, color, key, comm_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_comm_test_inter_( comm, flag, ierr )
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  *ierr = MPI_Comm_test_inter( MPI_Comm_f2c(*comm), flag );
}

void   mpi_comm_test_inter__( comm, flag, ierr )
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_comm_test_inter_( comm, flag, ierr );
}

void   MPI_COMM_TEST_INTER( comm, flag, ierr )
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_comm_test_inter_( comm, flag, ierr );
}

void   MPI_COMM_TEST_INTER_( comm, flag, ierr )
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_comm_test_inter_( comm, flag, ierr );
}

void   mpi_comm_test_inter( comm, flag, ierr )
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_comm_test_inter_( comm, flag, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_group_compare_( group1, group2, result, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  *ierr = MPI_Group_compare( MPI_Group_f2c(*group1), MPI_Group_f2c(*group2), result );
}

void   mpi_group_compare__( group1, group2, result, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  mpi_group_compare_( group1, group2, result, ierr );
}

void   MPI_GROUP_COMPARE( group1, group2, result, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  mpi_group_compare_( group1, group2, result, ierr );
}

void   MPI_GROUP_COMPARE_( group1, group2, result, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  mpi_group_compare_( group1, group2, result, ierr );
}


void   mpi_group_compare( group1, group2, result, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * result;
MPI_Fint *ierr;
{
  mpi_group_compare_( group1, group2, result, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_group_difference_( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Group_difference( MPI_Group_f2c(*group1), MPI_Group_f2c(*group2), 
	&local_group);
  *group_out = MPI_Group_c2f(local_group);
}

void   mpi_group_difference__( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_difference_( group1, group2, group_out, ierr );
}

void   MPI_GROUP_DIFFERENCE( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_difference_( group1, group2, group_out, ierr );
}

void   MPI_GROUP_DIFFERENCE_( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_difference_( group1, group2, group_out, ierr );
}

void   mpi_group_difference( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_difference_( group1, group2, group_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_group_excl_( group, n, ranks, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Group_excl( MPI_Group_f2c(*group), *n, ranks, &local_group );
  *newgroup = MPI_Group_c2f(local_group);
}

void   mpi_group_excl__( group, n, ranks, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_excl_( group, n, ranks, newgroup, ierr );
}

void   MPI_GROUP_EXCL( group, n, ranks, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_excl_( group, n, ranks, newgroup, ierr );
}

void   MPI_GROUP_EXCL_( group, n, ranks, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_excl_( group, n, ranks, newgroup, ierr );
}


void   mpi_group_excl( group, n, ranks, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_excl_( group, n, ranks, newgroup, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_group_free_( group, ierr)
MPI_Fint * group;
MPI_Fint *ierr;
{
  MPI_Group local_group = MPI_Group_f2c(*group);
  *ierr = MPI_Group_free( &local_group );
  *group = MPI_Group_c2f(local_group);
}

void   mpi_group_free__( group, ierr)
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_group_free_( group, ierr);
}

void   MPI_GROUP_FREE( group, ierr)
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_group_free_( group, ierr);
}

void   MPI_GROUP_FREE_( group, ierr)
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_group_free_( group, ierr);
}

void   mpi_group_free( group, ierr)
MPI_Fint * group;
MPI_Fint *ierr;
{
  mpi_group_free_( group, ierr);
}

/******************************************************/
/******************************************************/

void   mpi_group_incl_( group, n, ranks, group_out, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Group_incl( MPI_Group_f2c(*group), *n, ranks, &local_group );
  *group_out = MPI_Group_c2f(local_group);
}

void   mpi_group_incl__( group, n, ranks, group_out, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_incl_( group, n, ranks, group_out, ierr );
}

void   MPI_GROUP_INCL( group, n, ranks, group_out, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_incl_( group, n, ranks, group_out, ierr );
}

void   MPI_GROUP_INCL_( group, n, ranks, group_out, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_incl_( group, n, ranks, group_out, ierr );
}


void   mpi_group_incl( group, n, ranks, group_out, ierr )
MPI_Fint *group;
MPI_Fint *n;
MPI_Fint * ranks;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_incl_( group, n, ranks, group_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_group_intersection_( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Group_intersection( MPI_Group_f2c(*group1), MPI_Group_f2c(*group2), &local_group );
  *group_out = MPI_Group_c2f(local_group);
}

void   mpi_group_intersection__( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_intersection_( group1, group2, group_out, ierr );
}

void   MPI_GROUP_INTERSECTION( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_intersection_( group1, group2, group_out, ierr );
}

void   MPI_GROUP_INTERSECTION_( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_intersection_( group1, group2, group_out, ierr );
}

void   mpi_group_intersection( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_intersection_( group1, group2, group_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_group_rank_( group, rank, ierr)
MPI_Fint *group;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  *ierr = MPI_Group_rank( MPI_Group_f2c(*group), rank );
}

void   mpi_group_rank__( group, rank, ierr)
MPI_Fint *group;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_group_rank_( group, rank, ierr);
}

void   MPI_GROUP_RANK( group, rank, ierr)
MPI_Fint *group;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_group_rank_( group, rank, ierr);
}


void   MPI_GROUP_RANK_( group, rank, ierr)
MPI_Fint *group;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_group_rank_( group, rank, ierr);
}

void   mpi_group_rank( group, rank, ierr)
MPI_Fint *group;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_group_rank_( group, rank, ierr);
}

/******************************************************/
/******************************************************/

void   mpi_group_range_excl_( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Group_range_excl( MPI_Group_f2c(*group), *n, ranges, &local_group );
  *newgroup = MPI_Group_c2f(local_group);
}

void   mpi_group_range_excl__( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_range_excl_( group, n, ranges, newgroup, ierr ); 
}

void   MPI_GROUP_RANGE_EXCL( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_range_excl_( group, n, ranges, newgroup, ierr );
}

void   MPI_GROUP_RANGE_EXCL_( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_range_excl_( group, n, ranges, newgroup, ierr );
}

void   mpi_group_range_excl( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_range_excl_( group, n, ranges, newgroup, ierr );
}



/******************************************************/
/******************************************************/

void   mpi_group_range_incl_( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Group_range_incl( MPI_Group_f2c(*group), *n, ranges, &local_group );
  *newgroup = MPI_Group_c2f(local_group);
}

void   mpi_group_range_incl__( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_range_incl_( group, n, ranges, newgroup, ierr );
}

void   MPI_GROUP_RANGE_INCL( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_range_incl_( group, n, ranges, newgroup, ierr );
}

void   MPI_GROUP_RANGE_INCL_( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_range_incl_( group, n, ranges, newgroup, ierr );
}


void   mpi_group_range_incl( group, n, ranges, newgroup, ierr )
MPI_Fint *group;
MPI_Fint *n;
int ranges[][3];
MPI_Fint * newgroup;
MPI_Fint *ierr;
{
  mpi_group_range_incl_( group, n, ranges, newgroup, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_group_size_( group, size, ierr )
MPI_Fint *group;
MPI_Fint * size;
MPI_Fint *ierr;
{
  *ierr = MPI_Group_size( MPI_Group_f2c(*group), size );
}

void   mpi_group_size__( group, size, ierr )
MPI_Fint *group;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_group_size_( group, size, ierr );
}

void   MPI_GROUP_SIZE( group, size, ierr )
MPI_Fint *group;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_group_size_( group, size, ierr );
}

void   MPI_GROUP_SIZE_( group, size, ierr )
MPI_Fint *group;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_group_size_( group, size, ierr );
}

void   mpi_group_size( group, size, ierr )
MPI_Fint *group;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_group_size_( group, size, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_group_translate_ranks_( group_a, n, ranks_a, group_b, ranks_b, ierr)

MPI_Fint *group_a;
MPI_Fint *n;
MPI_Fint * ranks_a;
MPI_Fint *group_b;
MPI_Fint * ranks_b;
MPI_Fint *ierr;
{
  *ierr = MPI_Group_translate_ranks( MPI_Group_f2c(*group_a), *n, ranks_a, MPI_Group_f2c(*group_b), ranks_b );
}

void   mpi_group_translate_ranks__( group_a, n, ranks_a, group_b, ranks_b, ierr)
MPI_Fint *group_a;
MPI_Fint *n;
MPI_Fint * ranks_a;
MPI_Fint *group_b;
MPI_Fint * ranks_b;
MPI_Fint *ierr;
{
  mpi_group_translate_ranks_( group_a, n, ranks_a, group_b, ranks_b, ierr);
}

void   MPI_GROUP_TRANSLATE_RANKS( group_a, n, ranks_a, group_b, ranks_b, ierr)
MPI_Fint *group_a;
MPI_Fint *n;
MPI_Fint * ranks_a;
MPI_Fint *group_b;
MPI_Fint * ranks_b;
MPI_Fint *ierr;
{
  mpi_group_translate_ranks_( group_a, n, ranks_a, group_b, ranks_b, ierr);
}

void   MPI_GROUP_TRANSLATE_RANKS_( group_a, n, ranks_a, group_b, ranks_b, ierr)
MPI_Fint *group_a;
MPI_Fint *n;
MPI_Fint * ranks_a;
MPI_Fint *group_b;
MPI_Fint * ranks_b;
MPI_Fint *ierr;
{
  mpi_group_translate_ranks_( group_a, n, ranks_a, group_b, ranks_b, ierr);
}

void   mpi_group_translate_ranks( group_a, n, ranks_a, group_b, ranks_b, ierr)
MPI_Fint *group_a;
MPI_Fint *n;
MPI_Fint * ranks_a;
MPI_Fint *group_b;
MPI_Fint * ranks_b;
MPI_Fint *ierr;
{
  mpi_group_translate_ranks_( group_a, n, ranks_a, group_b, ranks_b, ierr);
}

/******************************************************/
/******************************************************/

void   mpi_group_union_( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  MPI_Group local_group;
  *ierr = MPI_Group_union( MPI_Group_f2c(*group1), MPI_Group_f2c(*group2), &local_group );
  *group_out = MPI_Group_c2f(local_group);
}

void   mpi_group_union__( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_union_( group1, group2, group_out, ierr );
}

void   MPI_GROUP_UNION( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_union_( group1, group2, group_out, ierr );
}

void   MPI_GROUP_UNION_( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_union_( group1, group2, group_out, ierr );
}



void   mpi_group_union( group1, group2, group_out, ierr )
MPI_Fint *group1;
MPI_Fint *group2;
MPI_Fint * group_out;
MPI_Fint *ierr;
{
  mpi_group_union_( group1, group2, group_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_intercomm_create_( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr )
MPI_Fint *local_comm;
MPI_Fint *local_leader;
MPI_Fint *peer_comm;
MPI_Fint *remote_leader;
MPI_Fint *tag;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  MPI_Comm local_comm_out;
  *ierr = MPI_Intercomm_create( MPI_Comm_f2c(*local_comm), *local_leader, MPI_Comm_f2c(*peer_comm), *remote_leader, *tag, &local_comm_out );
  *comm_out = MPI_Comm_c2f(local_comm_out);
}

void   mpi_intercomm_create__( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr )
MPI_Fint *local_comm;
MPI_Fint *local_leader;
MPI_Fint *peer_comm;
MPI_Fint *remote_leader;
MPI_Fint *tag;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_intercomm_create_( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr );
}

void   MPI_INTERCOMM_CREATE( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr )
MPI_Fint *local_comm;
MPI_Fint *local_leader;
MPI_Fint *peer_comm;
MPI_Fint *remote_leader;
MPI_Fint *tag;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_intercomm_create_( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr );
}

void   MPI_INTERCOMM_CREATE_( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr )
MPI_Fint *local_comm;
MPI_Fint *local_leader;
MPI_Fint *peer_comm;
MPI_Fint *remote_leader;
MPI_Fint *tag;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_intercomm_create_( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr );
}


void   mpi_intercomm_create( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr )
MPI_Fint *local_comm;
MPI_Fint *local_leader;
MPI_Fint *peer_comm;
MPI_Fint *remote_leader;
MPI_Fint *tag;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_intercomm_create_( local_comm, local_leader, peer_comm, remote_leader, tag, comm_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_intercomm_merge_( comm, high, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *high;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  MPI_Comm local_comm_out;
  *ierr = MPI_Intercomm_merge( MPI_Comm_f2c(*comm), *high, &local_comm_out );
  *comm_out = MPI_Comm_c2f(local_comm_out);
}

void   mpi_intercomm_merge__( comm, high, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *high;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_intercomm_merge_( comm, high, comm_out, ierr );
}

void   MPI_INTERCOMM_MERGE( comm, high, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *high;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_intercomm_merge_( comm, high, comm_out, ierr );
}

void   MPI_INTERCOMM_MERGE_( comm, high, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *high;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_intercomm_merge_( comm, high, comm_out, ierr );
}


void   mpi_intercomm_merge( comm, high, comm_out, ierr )
MPI_Fint *comm;
MPI_Fint *high;
MPI_Fint * comm_out;
MPI_Fint *ierr;
{
  mpi_intercomm_merge_( comm, high, comm_out, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_keyval_create_( copy_fn, delete_fn, keyval, extra_state, ierr )
MPI_Copy_function * copy_fn;
MPI_Delete_function * delete_fn;
MPI_Fint * keyval;
void * extra_state;
MPI_Fint *ierr;
{
  *ierr = MPI_Keyval_create( copy_fn, delete_fn, keyval, extra_state );
}

void   mpi_keyval_create__( copy_fn, delete_fn, keyval, extra_state, ierr )
MPI_Copy_function * copy_fn;
MPI_Delete_function * delete_fn;
MPI_Fint * keyval;
void * extra_state;
MPI_Fint *ierr;
{
  mpi_keyval_create_( copy_fn, delete_fn, keyval, extra_state, ierr );
}

void   MPI_KEYVAL_CREATE( copy_fn, delete_fn, keyval, extra_state, ierr )
MPI_Copy_function * copy_fn;
MPI_Delete_function * delete_fn;
MPI_Fint * keyval;
void * extra_state;
MPI_Fint *ierr;
{
  mpi_keyval_create_( copy_fn, delete_fn, keyval, extra_state, ierr );
}

void   MPI_KEYVAL_CREATE_( copy_fn, delete_fn, keyval, extra_state, ierr )
MPI_Copy_function * copy_fn;
MPI_Delete_function * delete_fn;
MPI_Fint * keyval;
void * extra_state;
MPI_Fint *ierr;
{
  mpi_keyval_create_( copy_fn, delete_fn, keyval, extra_state, ierr );
}

void   mpi_keyval_create( copy_fn, delete_fn, keyval, extra_state, ierr )
MPI_Copy_function * copy_fn;
MPI_Delete_function * delete_fn;
MPI_Fint * keyval;
void * extra_state;
MPI_Fint *ierr;
{
  mpi_keyval_create_( copy_fn, delete_fn, keyval, extra_state, ierr );
}


/******************************************************/
/******************************************************/

void   mpi_keyval_free_( keyval, ierr )
MPI_Fint * keyval;
MPI_Fint *ierr;
{
  *ierr = MPI_Keyval_free( keyval );
}

void   mpi_keyval_free__( keyval, ierr )
MPI_Fint * keyval;
MPI_Fint *ierr;
{
  mpi_keyval_free_( keyval, ierr );
}

void   MPI_KEYVAL_FREE( keyval, ierr )
MPI_Fint * keyval;
MPI_Fint *ierr;
{
  mpi_keyval_free_( keyval, ierr );
}

void   MPI_KEYVAL_FREE_( keyval, ierr )
MPI_Fint * keyval;
MPI_Fint *ierr;
{
  mpi_keyval_free_( keyval, ierr );
}


void   mpi_keyval_free( keyval, ierr )
MPI_Fint * keyval;
MPI_Fint *ierr;
{
  mpi_keyval_free_( keyval, ierr );
}


/******************************************************/
/******************************************************/

void  mpi_abort_( comm, errorcode , ierr)
MPI_Fint *comm;
MPI_Fint *errorcode;
MPI_Fint *ierr;
{
  *ierr = MPI_Abort( MPI_Comm_f2c(*comm), *errorcode );
}

void  mpi_abort__( comm, errorcode , ierr)
MPI_Fint *comm;
MPI_Fint *errorcode;
MPI_Fint *ierr;
{
  mpi_abort_( comm, errorcode , ierr);
}

void  MPI_ABORT( comm, errorcode , ierr)
MPI_Fint *comm;
MPI_Fint *errorcode;
MPI_Fint *ierr;
{
  mpi_abort_( comm, errorcode , ierr);
}

void  MPI_ABORT_( comm, errorcode , ierr)
MPI_Fint *comm;
MPI_Fint *errorcode;
MPI_Fint *ierr;
{
  mpi_abort_( comm, errorcode , ierr);
}

void  mpi_abort( comm, errorcode , ierr)
MPI_Fint *comm;
MPI_Fint *errorcode;
MPI_Fint *ierr;
{
  mpi_abort_( comm, errorcode , ierr);
}

/******************************************************/
/******************************************************/

void  mpi_error_class_( errorcode, errorclass, ierr )
MPI_Fint *errorcode;
MPI_Fint * errorclass;
MPI_Fint *ierr;
{
  *ierr = MPI_Error_class( *errorcode, errorclass );
}

void  mpi_error_class__( errorcode, errorclass, ierr )
MPI_Fint *errorcode;
MPI_Fint * errorclass;
MPI_Fint *ierr;
{
  mpi_error_class_( errorcode, errorclass, ierr );
}

void  MPI_ERROR_CLASS( errorcode, errorclass, ierr )
MPI_Fint *errorcode;
MPI_Fint * errorclass;
MPI_Fint *ierr;
{
  mpi_error_class_( errorcode, errorclass, ierr );
}

void  MPI_ERROR_CLASS_( errorcode, errorclass, ierr )
MPI_Fint *errorcode;
MPI_Fint * errorclass;
MPI_Fint *ierr;
{
  mpi_error_class_( errorcode, errorclass, ierr );
}

void  mpi_error_class( errorcode, errorclass, ierr )
MPI_Fint *errorcode;
MPI_Fint * errorclass;
MPI_Fint *ierr;
{
  mpi_error_class_( errorcode, errorclass, ierr );
}


/******************************************************/
/******************************************************/

void  mpi_errhandler_create_( function, errhandler , ierr)
MPI_Handler_function * function;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  *ierr = MPI_Errhandler_create( function, errhandler );
}

void  mpi_errhandler_create__( function, errhandler , ierr)
MPI_Handler_function * function;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_create_( function, errhandler , ierr);
}

void  MPI_ERRHANDLER_CREATE( function, errhandler , ierr)
MPI_Handler_function * function;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_create_( function, errhandler , ierr);
}


void  MPI_ERRHANDLER_CREATE_( function, errhandler , ierr)
MPI_Handler_function * function;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_create_( function, errhandler , ierr);
}

void  mpi_errhandler_create( function, errhandler , ierr)
MPI_Handler_function * function;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_create_( function, errhandler , ierr);
}


/******************************************************/
/******************************************************/

void  mpi_errhandler_free_( errhandler, ierr )
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  *ierr = MPI_Errhandler_free( errhandler );
}

void  mpi_errhandler_free__( errhandler, ierr )
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_free_( errhandler, ierr );
}

void  MPI_ERRHANDLER_FREE( errhandler, ierr )
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_free_( errhandler, ierr );
}

void  MPI_ERRHANDLER_FREE_( errhandler, ierr )
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_free_( errhandler, ierr );
}


void  mpi_errhandler_free( errhandler, ierr )
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_free_( errhandler, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_errhandler_get_( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  *ierr = MPI_Errhandler_get( MPI_Comm_f2c(*comm), errhandler );
}

void  mpi_errhandler_get__( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_get_( comm, errhandler, ierr );
}

void  MPI_ERRHANDLER_GET( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_get_( comm, errhandler, ierr );
}

void  MPI_ERRHANDLER_GET_( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_get_( comm, errhandler, ierr );
}


void  mpi_errhandler_get( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler * errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_get_( comm, errhandler, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_error_string_( errorcode, string, resultlen, ierr )
MPI_Fint *errorcode;
char * string;
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  *ierr = MPI_Error_string( *errorcode, string, resultlen );
}

void  mpi_error_string__( errorcode, string, resultlen, ierr )
MPI_Fint *errorcode;
char * string;
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  mpi_error_string_( errorcode, string, resultlen, ierr );
}

void  MPI_ERROR_STRING( errorcode, string, resultlen, ierr )
MPI_Fint *errorcode;
char * string;
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  mpi_error_string_( errorcode, string, resultlen, ierr );
}

void  MPI_ERROR_STRING_( errorcode, string, resultlen, ierr )
MPI_Fint *errorcode;
char * string;
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  mpi_error_string_( errorcode, string, resultlen, ierr );
}

void  mpi_error_string( errorcode, string, resultlen, ierr )
MPI_Fint *errorcode;
char * string;
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  mpi_error_string_( errorcode, string, resultlen, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_errhandler_set_( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler *errhandler;
MPI_Fint *ierr;
{
  *ierr = MPI_Errhandler_set( MPI_Comm_f2c(*comm), *errhandler );
}

void  mpi_errhandler_set__( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler *errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_set_( comm, errhandler, ierr );
}

void  MPI_ERRHANDLER_SET( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler *errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_set_( comm, errhandler, ierr );
}


void  MPI_ERRHANDLER_SET_( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler *errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_set_( comm, errhandler, ierr );
}

void  mpi_errhandler_set( comm, errhandler, ierr )
MPI_Fint *comm;
MPI_Errhandler *errhandler;
MPI_Fint *ierr;
{
  mpi_errhandler_set_( comm, errhandler, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_finalize_( ierr )
MPI_Fint *ierr;
{
  *ierr = MPI_Finalize(  );
}

void  mpi_finalize__( ierr )
MPI_Fint *ierr;
{
  mpi_finalize_( ierr );
}

void  MPI_FINALIZE( ierr )
MPI_Fint *ierr;
{
  mpi_finalize_( ierr );
}

void  MPI_FINALIZE_( ierr )
MPI_Fint *ierr;
{
  mpi_finalize_( ierr );
}

void  mpi_finalize( ierr )
MPI_Fint *ierr;
{
  mpi_finalize_( ierr );
}

/******************************************************/
/******************************************************/

void  mpi_get_processor_name_( name, resultlen, ierr )
char * name;
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  *ierr = MPI_Get_processor_name( name, resultlen );
}

void  mpi_get_processor_name__( name, resultlen, ierr )
char * name; 
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  mpi_get_processor_name_( name, resultlen, ierr );
}

void  MPI_GET_PROCESSOR_NAME( name, resultlen, ierr )
char * name; 
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  mpi_get_processor_name_( name, resultlen, ierr );
} 

void  MPI_GET_PROCESSOR_NAME_( name, resultlen, ierr )
char * name;
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  mpi_get_processor_name_( name, resultlen, ierr );
}

void  mpi_get_processor_name( name, resultlen, ierr )
char * name; 
MPI_Fint * resultlen;
MPI_Fint *ierr;
{
  mpi_get_processor_name_( name, resultlen, ierr );
} 

/******************************************************/
/******************************************************/

#ifndef TAU_WEAK_MPI_INIT
void  mpi_init_( ierr)
MPI_Fint *ierr; 
{
  *ierr = MPI_Init( 0, (char ***)0);
}

void  mpi_init__( ierr)
MPI_Fint *ierr;
{
  mpi_init_( ierr);
}

void  MPI_INIT( ierr)
MPI_Fint *ierr;
{
  mpi_init_( ierr);
}

void  MPI_INIT_( ierr)
MPI_Fint *ierr;
{
  mpi_init_( ierr);
}

void  mpi_init( ierr)
MPI_Fint *ierr;
{
  mpi_init_( ierr);
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPI_THREADED
void  mpi_init_thread_ (required, provided, ierr )
MPI_Fint *required;
MPI_Fint *provided;
MPI_Fint *ierr;
{
  *ierr = MPI_Init_thread( 0, (char ***)0, *required, provided );
}

void  mpi_init_thread__ (required, provided, ierr )
MPI_Fint *required;
MPI_Fint *provided;
MPI_Fint *ierr;
{
  mpi_init_thread_(required, provided, ierr );
}

void  MPI_INIT_THREAD(required, provided, ierr )
MPI_Fint *required;
MPI_Fint *provided;
MPI_Fint *ierr;
{
  mpi_init_thread_(required, provided, ierr );
} 

void  MPI_INIT_THREAD_(required, provided, ierr )
MPI_Fint *required;
MPI_Fint *provided;
MPI_Fint *ierr;
{
  mpi_init_thread_(required, provided, ierr );
}

void  mpi_init_thread(required, provided, ierr )
MPI_Fint *required;
MPI_Fint *provided;
MPI_Fint *ierr;
{
  mpi_init_thread_(required, provided, ierr );
} 

#endif /* TAU_MPI_THREADED */
#endif /* TAU_WEAK_MPI_INIT */

/******************************************************/
/******************************************************/

double  mpi_wtick_( )
{
  return MPI_Wtick(  );
}

double  mpi_wtick__( )
{
  return MPI_Wtick(  );
}

double  MPI_WTICK( )
{
  return MPI_Wtick(  );
}

double  MPI_WTICK_( )
{
  return MPI_Wtick(  );
}

double  mpi_wtick( )
{
  return MPI_Wtick(  );
}

/******************************************************/
/******************************************************/

double  mpi_wtime_(  )
{
  return MPI_Wtime(  );
}

double  mpi_wtime__(  )
{
  return MPI_Wtime(  );
}

double  MPI_WTIME(  )
{
  return MPI_Wtime(  );
}

double  MPI_WTIME_(  )
{
  return MPI_Wtime(  );
}

double  mpi_wtime(  )
{
  return MPI_Wtime(  );
}


/******************************************************/
/******************************************************/

void  mpi_address_( location, address , ierr)
void * location;
MPI_Fint * address;
MPI_Fint *ierr;
{
  MPI_Aint c_address;
  *ierr = MPI_Address( location, &c_address );
  *address = c_address;
}

void  mpi_address__( location, address , ierr)
void * location;
MPI_Fint * address;
MPI_Fint *ierr;
{
  mpi_address_( location, address , ierr);
}

void  MPI_ADDRESS( location, address , ierr)
void * location;
MPI_Fint * address;
MPI_Fint *ierr;
{
  mpi_address_( location, address , ierr);
}

void  MPI_ADDRESS_( location, address , ierr)
void * location;
MPI_Fint * address;
MPI_Fint *ierr;
{
  mpi_address_( location, address , ierr);
}

void  mpi_address( location, address , ierr)
void * location;
MPI_Fint * address;
MPI_Fint *ierr;
{
  mpi_address_( location, address , ierr);
}

/******************************************************/
/******************************************************/
void  mpi_bsend_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Bsend( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm) );
}

void  mpi_bsend__( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_bsend_( buf, count, datatype, dest, tag, comm, ierr );
}

void  MPI_BSEND( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_bsend_( buf, count, datatype, dest, tag, comm, ierr );
}

void  MPI_BSEND_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_bsend_( buf, count, datatype, dest, tag, comm, ierr );
}

void  mpi_bsend( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_bsend_( buf, count, datatype, dest, tag, comm, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_bsend_init_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Bsend_init( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_bsend_init__( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_bsend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_BSEND_INIT( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_bsend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_BSEND_INIT_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_bsend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  mpi_bsend_init( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_bsend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_buffer_attach_( buffer, size, ierr )
void * buffer;
MPI_Fint *size;
MPI_Fint *ierr;
{
  *ierr = MPI_Buffer_attach( buffer, *size );
}

void  mpi_buffer_attach__( buffer, size, ierr )
void * buffer;
MPI_Fint *size;
MPI_Fint *ierr;
{
  mpi_buffer_attach_( buffer, size, ierr );
}

void  MPI_BUFFER_ATTACH( buffer, size, ierr )
void * buffer;
MPI_Fint *size;
MPI_Fint *ierr;
{
  mpi_buffer_attach_( buffer, size, ierr );
}

void  MPI_BUFFER_ATTACH_( buffer, size, ierr )
void * buffer;
MPI_Fint *size;
MPI_Fint *ierr;
{
  mpi_buffer_attach_( buffer, size, ierr );
}


void  mpi_buffer_attach( buffer, size, ierr )
void * buffer;
MPI_Fint *size;
MPI_Fint *ierr;
{
  mpi_buffer_attach_( buffer, size, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_buffer_detach_( buffer, size, ierr )
void * buffer;
MPI_Fint * size;
MPI_Fint *ierr;
{
  *ierr = MPI_Buffer_detach( buffer, size );
}

void  mpi_buffer_detach__( buffer, size, ierr )
void * buffer;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_buffer_detach_( buffer, size, ierr );
}

void  MPI_BUFFER_DETACH( buffer, size, ierr )
void * buffer;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_buffer_detach_( buffer, size, ierr );
}

void  MPI_BUFFER_DETACH_( buffer, size, ierr )
void * buffer;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_buffer_detach_( buffer, size, ierr );
}

void  mpi_buffer_detach( buffer, size, ierr )
void * buffer;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_buffer_detach_( buffer, size, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_cancel_( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request = MPI_Request_f2c(*request);
  *ierr = MPI_Cancel( &local_request );
}

void  mpi_cancel__( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_cancel_( request, ierr );
}

void  MPI_CANCEL( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_cancel_( request, ierr );
}

void  MPI_CANCEL_( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_cancel_( request, ierr );
}


void  mpi_cancel( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_cancel_( request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_request_free_( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request = MPI_Request_f2c(*request);
  *ierr = MPI_Request_free( &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_request_free__( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_request_free_( request, ierr );
}

void  MPI_REQUEST_FREE( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_request_free_( request, ierr );
}

void  MPI_REQUEST_FREE_( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_request_free_( request, ierr );
}


void  mpi_request_free( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_request_free_( request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_recv_init_( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Recv_init( buf, *count, MPI_Type_f2c(*datatype), *source, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_recv_init__( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_recv_init_( buf, count, datatype, source, tag, comm, request, ierr );
}

void  MPI_RECV_INIT( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_recv_init_( buf, count, datatype, source, tag, comm, request, ierr );
}

void  MPI_RECV_INIT_( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_recv_init_( buf, count, datatype, source, tag, comm, request, ierr );
}

void  mpi_recv_init( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_recv_init_( buf, count, datatype, source, tag, comm, request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_send_init_( buf, count, datatype, dest, tag, comm, request, ierr )	
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Send_init( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}


void  mpi_send_init__( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_send_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_SEND_INIT( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_send_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_SEND_INIT_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_send_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  mpi_send_init( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_send_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_get_elements_( status, datatype, elements, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * elements;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  MPI_Status_f2c(status, &local_status);
  *ierr = MPI_Get_elements( &local_status, MPI_Type_f2c(*datatype), elements );
}

void   mpi_get_elements__( status, datatype, elements, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * elements;
MPI_Fint *ierr;
{
  mpi_get_elements_( status, datatype, elements, ierr );
}

void   MPI_GET_ELEMENTS( status, datatype, elements, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * elements;
MPI_Fint *ierr;
{
  mpi_get_elements_( status, datatype, elements, ierr );
}

void   MPI_GET_ELEMENTS_( status, datatype, elements, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * elements;
MPI_Fint *ierr;
{
  mpi_get_elements_( status, datatype, elements, ierr );
}


void   mpi_get_elements( status, datatype, elements, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * elements;
MPI_Fint *ierr;
{
  mpi_get_elements_( status, datatype, elements, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_get_count_( status, datatype, count, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * count;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  MPI_Status_f2c(status, &local_status);
  *ierr = MPI_Get_count( &local_status, MPI_Type_f2c(*datatype), count );
}

void  mpi_get_count__( status, datatype, count, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * count;
MPI_Fint *ierr;
{
  mpi_get_count_( status, datatype, count, ierr );
}

void  MPI_GET_COUNT( status, datatype, count, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * count;
MPI_Fint *ierr;
{
  mpi_get_count_( status, datatype, count, ierr );
}

void  MPI_GET_COUNT_( status, datatype, count, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * count;
MPI_Fint *ierr;
{
  mpi_get_count_( status, datatype, count, ierr );
}

void  mpi_get_count( status, datatype, count, ierr )
MPI_Fint * status;
MPI_Fint *datatype;
MPI_Fint * count;
MPI_Fint *ierr;
{
  mpi_get_count_( status, datatype, count, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_ibsend_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Ibsend( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_ibsend__( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_ibsend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_IBSEND( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_ibsend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_IBSEND_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_ibsend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  mpi_ibsend( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_ibsend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_iprobe_( source, tag, comm, flag, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  *ierr = MPI_Iprobe( *source, *tag, MPI_Comm_f2c(*comm), flag, &local_status );
  MPI_Status_c2f(&local_status, status);
}

void  mpi_iprobe__( source, tag, comm, flag, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_iprobe_( source, tag, comm, flag, status, ierr );
}

void  MPI_IPROBE( source, tag, comm, flag, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_iprobe_( source, tag, comm, flag, status, ierr );
}

void  MPI_IPROBE_( source, tag, comm, flag, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_iprobe_( source, tag, comm, flag, status, ierr );
}



void  mpi_iprobe( source, tag, comm, flag, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_iprobe_( source, tag, comm, flag, status, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_irecv_( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Irecv( buf, *count, MPI_Type_f2c(*datatype), *source, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_irecv__( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_irecv_( buf, count, datatype, source, tag, comm, request, ierr );
}

void  MPI_IRECV( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_irecv_( buf, count, datatype, source, tag, comm, request, ierr );
}

void  MPI_IRECV_( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_irecv_( buf, count, datatype, source, tag, comm, request, ierr );
}


void  mpi_irecv( buf, count, datatype, source, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_irecv_( buf, count, datatype, source, tag, comm, request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_irsend_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Irsend( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_irsend__( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_irsend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_IRSEND( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_irsend_( buf, count, datatype, dest, tag, comm, request, ierr );
} 


void  MPI_IRSEND_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_irsend_( buf, count, datatype, dest, tag, comm, request, ierr );
}


void  mpi_irsend( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_irsend_( buf, count, datatype, dest, tag, comm, request, ierr );
} 

/******************************************************/
/******************************************************/

void  mpi_isend_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Isend( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_isend__( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_isend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_ISEND( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_isend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_ISEND_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_isend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  mpi_isend( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_isend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_issend_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Issend( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_issend__( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_issend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_ISSEND( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_issend_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_ISSEND_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_issend_( buf, count, datatype, dest, tag, comm, request, ierr );
}


void  mpi_issend( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_issend_( buf, count, datatype, dest, tag, comm, request, ierr );
}


/******************************************************/
/******************************************************/

void   mpi_pack_( inbuf, incount, type, outbuf, outcount, position, comm, ierr )
void * inbuf;
MPI_Fint *incount;
MPI_Fint *type;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint * position;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Pack( inbuf, *incount, MPI_Type_f2c(*type), outbuf, *outcount, position, MPI_Comm_f2c(*comm) );
}

void   mpi_pack__( inbuf, incount, type, outbuf, outcount, position, comm, ierr )
void * inbuf;
MPI_Fint *incount;
MPI_Fint *type;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint * position;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_pack_( inbuf, incount, type, outbuf, outcount, position, comm, ierr );
}

void   MPI_PACK( inbuf, incount, type, outbuf, outcount, position, comm, ierr )
void * inbuf;
MPI_Fint *incount;
MPI_Fint *type;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint * position;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_pack_( inbuf, incount, type, outbuf, outcount, position, comm, ierr );
}

void   MPI_PACK_( inbuf, incount, type, outbuf, outcount, position, comm, ierr )
void * inbuf;
MPI_Fint *incount;
MPI_Fint *type;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint * position;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_pack_( inbuf, incount, type, outbuf, outcount, position, comm, ierr );
}


void   mpi_pack( inbuf, incount, type, outbuf, outcount, position, comm, ierr )
void * inbuf;
MPI_Fint *incount;
MPI_Fint *type;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint * position;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_pack_( inbuf, incount, type, outbuf, outcount, position, comm, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_pack_size_( incount, datatype, comm, size, ierr )
MPI_Fint *incount;
MPI_Fint *datatype;
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  *ierr = MPI_Pack_size( *incount, MPI_Type_f2c(*datatype), MPI_Comm_f2c(*comm), size );
}

void   mpi_pack_size__( incount, datatype, comm, size, ierr )
MPI_Fint *incount;
MPI_Fint *datatype;
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_pack_size_( incount, datatype, comm, size, ierr );
}

void   MPI_PACK_SIZE( incount, datatype, comm, size, ierr )
MPI_Fint *incount;
MPI_Fint *datatype;
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_pack_size_( incount, datatype, comm, size, ierr );
}

void   MPI_PACK_SIZE_( incount, datatype, comm, size, ierr )
MPI_Fint *incount;
MPI_Fint *datatype;
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_pack_size_( incount, datatype, comm, size, ierr );
}

void   mpi_pack_size( incount, datatype, comm, size, ierr )
MPI_Fint *incount;
MPI_Fint *datatype;
MPI_Fint *comm;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_pack_size_( incount, datatype, comm, size, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_probe_( source, tag, comm, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Status local_status; 
  *ierr = MPI_Probe( *source, *tag, MPI_Comm_f2c(*comm), &local_status );
  MPI_Status_c2f(&local_status, status);
}

void  mpi_probe__( source, tag, comm, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_probe_( source, tag, comm, status, ierr );
}

void  MPI_PROBE( source, tag, comm, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_probe_( source, tag, comm, status, ierr );
}

void  MPI_PROBE_( source, tag, comm, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_probe_( source, tag, comm, status, ierr );
}


void  mpi_probe( source, tag, comm, status, ierr )
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_probe_( source, tag, comm, status, ierr );
}


/******************************************************/
/******************************************************/

void  mpi_recv_( buf, count, datatype, source, tag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Status s;
  *ierr = MPI_Recv( buf, *count, MPI_Type_f2c(*datatype), *source, *tag, MPI_Comm_f2c(*comm), &s );
  MPI_Status_c2f(&s, status);

}

void  mpi_recv__( buf, count, datatype, source, tag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_recv_( buf, count, datatype, source, tag, comm, status, ierr );
}

void  MPI_RECV( buf, count, datatype, source, tag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_recv_( buf, count, datatype, source, tag, comm, status, ierr );
}


void  MPI_RECV_( buf, count, datatype, source, tag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_recv_( buf, count, datatype, source, tag, comm, status, ierr );
}

void  mpi_recv( buf, count, datatype, source, tag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *source;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_recv_( buf, count, datatype, source, tag, comm, status, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_rsend_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Rsend( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm) );
}

void  mpi_rsend__( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_rsend_( buf, count, datatype, dest, tag, comm, ierr );
}

void  MPI_RSEND( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_rsend_( buf, count, datatype, dest, tag, comm, ierr );
}

void  MPI_RSEND_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_rsend_( buf, count, datatype, dest, tag, comm, ierr );
}

void  mpi_rsend( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_rsend_( buf, count, datatype, dest, tag, comm, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_rsend_init_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Rsend_init( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_rsend_init__( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_rsend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_RSEND_INIT( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_rsend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_RSEND_INIT_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_rsend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}


void  mpi_rsend_init( buf, count, datatype, dest, tag, comm, request, ierr ) 
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_rsend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

/******************************************************/
/******************************************************/
void  mpi_send_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Send( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm ));
}
void  mpi_send__( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_send_( buf, count, datatype, dest, tag, comm, ierr );
}

void  MPI_SEND( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_send_( buf, count, datatype, dest, tag, comm, ierr );
}

void  MPI_SEND_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_send_( buf, count, datatype, dest, tag, comm, ierr );
}

void  mpi_send( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_send_( buf, count, datatype, dest, tag, comm, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_sendrecv_( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr )
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
MPI_Fint *dest;
MPI_Fint *sendtag;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  *ierr = MPI_Sendrecv( sendbuf, *sendcount, MPI_Type_f2c(*sendtype), *dest, *sendtag, recvbuf, *recvcount, MPI_Type_f2c(*recvtype), *source, *recvtag, MPI_Comm_f2c(*comm), &local_status );
  MPI_Status_c2f(&local_status, status);
}

void  mpi_sendrecv__( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr )
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
MPI_Fint *dest;
MPI_Fint *sendtag;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_sendrecv_( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr );
}

void  MPI_SENDRECV( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr )
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
MPI_Fint *dest;
MPI_Fint *sendtag;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_sendrecv_( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr );
}

void  MPI_SENDRECV_( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr )
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
MPI_Fint *dest;
MPI_Fint *sendtag;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_sendrecv_( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr );
}


void  mpi_sendrecv( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr )
void * sendbuf;
MPI_Fint *sendcount;
MPI_Fint *sendtype;
MPI_Fint *dest;
MPI_Fint *sendtag;
void * recvbuf;
MPI_Fint *recvcount;
MPI_Fint *recvtype;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_sendrecv_( sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr );
}

/******************************************************/
/******************************************************/
void  mpi_sendrecv_replace_( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *sendtag;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  *ierr = MPI_Sendrecv_replace( buf, *count, MPI_Type_f2c(*datatype), *dest, *sendtag, *source, *recvtag, MPI_Comm_f2c(*comm), &local_status );
  MPI_Status_c2f(&local_status, status);
}

void  mpi_sendrecv_replace__( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *sendtag;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_sendrecv_replace_( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr );
}

void  MPI_SENDRECV_REPLACE( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *sendtag;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_sendrecv_replace_( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr );
}

void  MPI_SENDRECV_REPLACE_( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *sendtag;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_sendrecv_replace_( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr );
}


void  mpi_sendrecv_replace( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *sendtag;
MPI_Fint *source;
MPI_Fint *recvtag;
MPI_Fint *comm;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_sendrecv_replace_( buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_ssend_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Ssend( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm) );
}

void  mpi_ssend__( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_ssend_( buf, count, datatype, dest, tag, comm, ierr );
}

void  MPI_SSEND( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_ssend_( buf, count, datatype, dest, tag, comm, ierr );
}

void  MPI_SSEND_( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_ssend_( buf, count, datatype, dest, tag, comm, ierr );
}


void  mpi_ssend( buf, count, datatype, dest, tag, comm, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_ssend_( buf, count, datatype, dest, tag, comm, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_ssend_init_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request;
  *ierr = MPI_Ssend_init( buf, *count, MPI_Type_f2c(*datatype), *dest, *tag, MPI_Comm_f2c(*comm), &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_ssend_init__( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_ssend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_SSEND_INIT( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_ssend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

void  MPI_SSEND_INIT_( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_ssend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}


void  mpi_ssend_init( buf, count, datatype, dest, tag, comm, request, ierr )
void * buf;
MPI_Fint *count;
MPI_Fint *datatype;
MPI_Fint *dest;
MPI_Fint *tag;
MPI_Fint *comm;
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_ssend_init_( buf, count, datatype, dest, tag, comm, request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_start_( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  MPI_Request local_request = MPI_Request_f2c(*request);
  *ierr = MPI_Start( &local_request );
  *request = TAU_MPI_Request_c2f(local_request);
}

void  mpi_start__( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_start_( request, ierr );
}

void  MPI_START( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_start_( request, ierr );
}

void  MPI_START_( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_start_( request, ierr );
}


void  mpi_start( request, ierr )
MPI_Fint * request;
MPI_Fint *ierr;
{
  mpi_start_( request, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_startall_( count, array_of_requests, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint *ierr;
{
  TAU_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *count);
  TAU_ASSIGN_VALUES(local_requests, array_of_requests, *count, MPI_Request_f2c);
  *ierr = MPI_Startall( *count, local_requests );
  TAU_ASSIGN_VALUES(array_of_requests, local_requests, *count, TAU_MPI_Request_c2f);
  TAU_FREE_LOCAL(local_requests);
}

void  mpi_startall__( count, array_of_requests, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint *ierr;
{
  mpi_startall_( count, array_of_requests, ierr );
}

void  MPI_STARTALL( count, array_of_requests, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint *ierr;
{
  mpi_startall_( count, array_of_requests, ierr );
}

void  MPI_STARTALL_( count, array_of_requests, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint *ierr;
{
  mpi_startall_( count, array_of_requests, ierr );
}

void  mpi_startall( count, array_of_requests, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint *ierr;
{
  mpi_startall_( count, array_of_requests, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_test_( request, flag, status, ierr )
MPI_Fint * request;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  MPI_Request local_request = MPI_Request_f2c(*request);
  *ierr = MPI_Test( &local_request, flag, &local_status );
  *request = TAU_MPI_Request_c2f(local_request);
  MPI_Status_c2f(&local_status, status);
}

void   mpi_test__( request, flag, status, ierr )
MPI_Fint * request;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_test_( request, flag, status, ierr );
}

void   MPI_TEST( request, flag, status, ierr )
MPI_Fint * request;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_test_( request, flag, status, ierr );
}

void   MPI_TEST_( request, flag, status, ierr )
MPI_Fint * request;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_test_( request, flag, status, ierr );
}

void   mpi_test( request, flag, status, ierr )
MPI_Fint * request;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_test_( request, flag, status, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_testall_( count, array_of_requests, flag, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * flag;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  TAU_DECL_LOCAL(MPI_Status, local_statuses);
  TAU_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *count);
  TAU_ALLOC_LOCAL(MPI_Status, local_statuses, *count);
  TAU_ASSIGN_VALUES(local_requests, array_of_requests, *count, MPI_Request_f2c);
  TAU_ASSIGN_STATUS_F2C(local_statuses, array_of_statuses, *count, MPI_Status_f2c);
  *ierr = MPI_Testall( *count, local_requests, flag, local_statuses );
  TAU_ASSIGN_VALUES(array_of_requests, local_requests, *count, TAU_MPI_Request_c2f);
  TAU_ASSIGN_STATUS_C2F(array_of_statuses, local_statuses, *count, MPI_Status_c2f);
  TAU_FREE_LOCAL(local_requests);
  TAU_FREE_LOCAL(local_statuses);
}

void  mpi_testall__( count, array_of_requests, flag, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * flag;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_testall_( count, array_of_requests, flag, array_of_statuses, ierr );
}

void  MPI_TESTALL( count, array_of_requests, flag, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * flag;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_testall_( count, array_of_requests, flag, array_of_statuses, ierr );
}

void  MPI_TESTALL_( count, array_of_requests, flag, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * flag;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_testall_( count, array_of_requests, flag, array_of_statuses, ierr );
}


void  mpi_testall( count, array_of_requests, flag, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * flag;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_testall_( count, array_of_requests, flag, array_of_statuses, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_testany_( count, array_of_requests, index, flag, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  TAU_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *count);
  TAU_ASSIGN_VALUES(local_requests, array_of_requests, *count, MPI_Request_f2c);
  *ierr	 = MPI_Testany( *count, local_requests, index, flag, &local_status );
  TAU_ASSIGN_VALUES(array_of_requests, local_requests, *count, TAU_MPI_Request_c2f);
  MPI_Status_c2f(&local_status, status);
  TAU_FREE_LOCAL(local_requests);
  /* Increment the C index before returning it as a Fortran index as
     [0..N-1] => [1..N] array indexing differs in C and Fortran */
  (*index)++;
}

void  mpi_testany__( count, array_of_requests, index, flag, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_testany_( count, array_of_requests, index, flag, status, ierr );
}

void  MPI_TESTANY( count, array_of_requests, index, flag, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_testany_( count, array_of_requests, index, flag, status, ierr );
}

void  MPI_TESTANY_( count, array_of_requests, index, flag, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_testany_( count, array_of_requests, index, flag, status, ierr );
}


void  mpi_testany( count, array_of_requests, index, flag, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * flag;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_testany_( count, array_of_requests, index, flag, status, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_test_cancelled_( status, flag, ierr )
MPI_Fint * status;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  MPI_Status_f2c(status, &local_status);
  *ierr = MPI_Test_cancelled( &local_status, flag );
}

void  mpi_test_cancelled__( status, flag, ierr )
MPI_Fint * status;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_test_cancelled_( status, flag, ierr );
}

void  MPI_TEST_CANCELLED( status, flag, ierr )
MPI_Fint * status;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_test_cancelled_( status, flag, ierr );
}

void  MPI_TEST_CANCELLED_( status, flag, ierr )
MPI_Fint * status;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_test_cancelled_( status, flag, ierr );
}


void  mpi_test_cancelled( status, flag, ierr )
MPI_Fint * status;
MPI_Fint * flag;
MPI_Fint *ierr;
{
  mpi_test_cancelled_( status, flag, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_testsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )
MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  int i;
  TAU_DECL_LOCAL(MPI_Status, local_statuses);
  TAU_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *incount);
  TAU_ALLOC_LOCAL(MPI_Status, local_statuses, *incount);
  TAU_ASSIGN_VALUES(local_requests, array_of_requests, *incount, MPI_Request_f2c);
  TAU_ASSIGN_STATUS_F2C(local_statuses, array_of_statuses, *incount, MPI_Status_f2c);
  *ierr = MPI_Testsome( *incount, local_requests, outcount, array_of_indices, local_statuses );
  TAU_ASSIGN_VALUES(array_of_requests, local_requests, *outcount, TAU_MPI_Request_c2f);
  TAU_ASSIGN_STATUS_C2F(array_of_statuses, local_statuses, *outcount, MPI_Status_c2f);
  TAU_FREE_LOCAL(local_requests);
  TAU_FREE_LOCAL(local_statuses);
  /* Increment the C index before returning it as a Fortran index as
     [0..N-1] => [1..N] array indexing differs in C and Fortran */
  for (i=0; i < *outcount; i++) array_of_indices[i]++;
  
}

void  mpi_testsome__( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )
MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_testsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr );
}

void  MPI_TESTSOME( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )
MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_testsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr );
}

void  MPI_TESTSOME_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )
MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_testsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr );
}

void  mpi_testsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )
MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_testsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_type_commit_( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  MPI_Datatype local_data_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_Type_commit( &local_data_type);
  *datatype = MPI_Type_c2f(local_data_type);
}

void   mpi_type_commit__( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  mpi_type_commit_( datatype, ierr );
}

void   MPI_TYPE_COMMIT( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  mpi_type_commit_( datatype, ierr );
} 

void   MPI_TYPE_COMMIT_( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  mpi_type_commit_( datatype, ierr );
}


void   mpi_type_commit( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  mpi_type_commit_( datatype, ierr );
} 

/******************************************************/
/******************************************************/

void  mpi_type_contiguous_( count, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  MPI_Datatype local_new_type; 
  *ierr = MPI_Type_contiguous( *count, MPI_Type_f2c(*old_type), 
		&local_new_type );
  *newtype = MPI_Type_c2f(local_new_type);
}

void  mpi_type_contiguous__( count, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_contiguous_( count, old_type, newtype, ierr );
}

void  MPI_TYPE_CONTIGUOUS( count, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_contiguous_( count, old_type, newtype, ierr );
} 

void  MPI_TYPE_CONTIGUOUS_( count, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_contiguous_( count, old_type, newtype, ierr );
}

void  mpi_type_contiguous( count, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_contiguous_( count, old_type, newtype, ierr );
} 

/******************************************************/
/******************************************************/

void  mpi_type_extent_( datatype, extent, ierr )
MPI_Fint *datatype;
MPI_Fint * extent;
MPI_Fint *ierr;
{
  MPI_Aint c_extent;
  *ierr = MPI_Type_extent( MPI_Type_f2c(*datatype), &c_extent );
  *extent = c_extent;
}

void  mpi_type_extent__( datatype, extent, ierr )
MPI_Fint *datatype;
MPI_Fint * extent;
MPI_Fint *ierr;
{
  mpi_type_extent_( datatype, extent, ierr );
}

void  MPI_TYPE_EXTENT( datatype, extent, ierr )
MPI_Fint *datatype;
MPI_Fint * extent;
MPI_Fint *ierr;
{
  mpi_type_extent_( datatype, extent, ierr );
}

void  MPI_TYPE_EXTENT_( datatype, extent, ierr )
MPI_Fint *datatype;
MPI_Fint * extent;
MPI_Fint *ierr;
{
  mpi_type_extent_( datatype, extent, ierr );
}

void  mpi_type_extent( datatype, extent, ierr )
MPI_Fint *datatype;
MPI_Fint * extent;
MPI_Fint *ierr;
{
  mpi_type_extent_( datatype, extent, ierr );
}

/******************************************************/
/******************************************************/
void   mpi_type_free_( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  MPI_Datatype local_data_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_Type_free( &local_data_type );
  *datatype = MPI_Type_c2f(local_data_type);
}

void   mpi_type_free__( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  mpi_type_free_( datatype, ierr );
}

void   MPI_TYPE_FREE( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  mpi_type_free_( datatype, ierr );
} 

void   MPI_TYPE_FREE_( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  mpi_type_free_( datatype, ierr );
}


void   mpi_type_free( datatype, ierr )
MPI_Fint * datatype;
MPI_Fint *ierr;
{
  mpi_type_free_( datatype, ierr );
} 

/******************************************************/
/******************************************************/

void  mpi_type_hindexed_( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  MPI_Aint *c_indices;
  int i;
  MPI_Datatype local_new_type;

  /* We allocate an array of MPI_Aint and copy the MPIFint's into it
     We must do this because the C binding of MPI-1 uses MPI_Aint's which
     are the size of pointers, whereas the Fortran binding always uses 32-bit
     integers */
  c_indices = (MPI_Aint *) malloc (*count * sizeof(MPI_Aint));
  for (i=0; i < *count; i++) {
    c_indices[i] = indices[i];
  }

  *ierr = MPI_Type_hindexed( *count, blocklens, c_indices, MPI_Type_f2c(*old_type), &local_new_type );
  *newtype = MPI_Type_c2f(local_new_type);

  free (c_indices);
}

void  mpi_type_hindexed__( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_hindexed_( count, blocklens, indices, old_type, newtype, ierr );
}

void  MPI_TYPE_HINDEXED( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_hindexed_( count, blocklens, indices, old_type, newtype, ierr );
}

void  MPI_TYPE_HINDEXED_( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_hindexed_( count, blocklens, indices, old_type, newtype, ierr );
}


void  mpi_type_hindexed( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_hindexed_( count, blocklens, indices, old_type, newtype, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_type_hvector_( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  MPI_Datatype local_new_type;
  *ierr = MPI_Type_hvector( *count, *blocklen, *stride, MPI_Type_f2c(*old_type), &local_new_type );
  *newtype = MPI_Type_c2f(local_new_type);

}

void  mpi_type_hvector__( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_hvector_( count, blocklen, stride, old_type, newtype, ierr );
}

void  MPI_TYPE_HVECTOR( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_hvector_( count, blocklen, stride, old_type, newtype, ierr );
} 

void  MPI_TYPE_HVECTOR_( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_hvector_( count, blocklen, stride, old_type, newtype, ierr );
}

void  mpi_type_hvector( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_hvector_( count, blocklen, stride, old_type, newtype, ierr );
} 

/******************************************************/
/******************************************************/

void  mpi_type_indexed_( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  MPI_Datatype local_new_type;
  *ierr = MPI_Type_indexed( *count, blocklens, indices, MPI_Type_f2c(*old_type), &local_new_type );
  *newtype = MPI_Type_c2f(local_new_type);
}

void  mpi_type_indexed__( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_indexed_( count, blocklens, indices, old_type, newtype, ierr );
}

void  MPI_TYPE_INDEXED( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_indexed_( count, blocklens, indices, old_type, newtype, ierr );
} 

void  MPI_TYPE_INDEXED_( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_indexed_( count, blocklens, indices, old_type, newtype, ierr );
}

void  mpi_type_indexed( count, blocklens, indices, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_indexed_( count, blocklens, indices, old_type, newtype, ierr );
} 

/******************************************************/
/******************************************************/

void   mpi_type_lb_( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint *displacement;
MPI_Fint *ierr;
{
  MPI_Aint c_displacement;
  *ierr = MPI_Type_lb( MPI_Type_f2c(*datatype), &c_displacement );
  *displacement = c_displacement;
}

void   mpi_type_lb__( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint * displacement;
MPI_Fint *ierr;
{
  mpi_type_lb_( datatype, displacement, ierr );
}

void   MPI_TYPE_LB( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint * displacement;
MPI_Fint *ierr;
{
  mpi_type_lb_( datatype, displacement, ierr );
}

void   MPI_TYPE_LB_( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint * displacement;
MPI_Fint *ierr;
{
  mpi_type_lb_( datatype, displacement, ierr );
}


void   mpi_type_lb( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint * displacement;
MPI_Fint *ierr;
{
  mpi_type_lb_( datatype, displacement, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_type_size_( datatype, size, ierr )
MPI_Fint *datatype;
MPI_Fint * size;
MPI_Fint *ierr;
{
  *ierr = MPI_Type_size( MPI_Type_f2c(*datatype), size );
}

void   mpi_type_size__( datatype, size, ierr )
MPI_Fint *datatype;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_type_size_( datatype, size, ierr );
}

void   MPI_TYPE_SIZE( datatype, size, ierr )
MPI_Fint *datatype;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_type_size_( datatype, size, ierr );
}

void   MPI_TYPE_SIZE_( datatype, size, ierr )
MPI_Fint *datatype;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_type_size_( datatype, size, ierr );
}


void   mpi_type_size( datatype, size, ierr )
MPI_Fint *datatype;
MPI_Fint * size;
MPI_Fint *ierr;
{
  mpi_type_size_( datatype, size, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_type_struct_( count, blocklens, indices, old_types, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint * old_types;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  MPI_Aint *c_indices;
  int i;
  MPI_Datatype local_new_type; 
  TAU_DECL_ALLOC_LOCAL(MPI_Datatype, local_types, *count);
  TAU_ASSIGN_VALUES(local_types, old_types, *count, MPI_Type_f2c);

  /* We allocate an array of MPI_Aint and copy the MPIFint's into it
     We must do this because the C binding of MPI-1 uses MPI_Aint's which
     are the size of pointers, whereas the Fortran binding always uses 32-bit
     integers */
  c_indices = (MPI_Aint *) malloc (*count * sizeof(MPI_Aint));
  for (i=0; i < *count; i++) {
    c_indices[i] = indices[i];
  }

  *ierr = MPI_Type_struct( *count, blocklens, c_indices, local_types, &local_new_type );
  TAU_FREE_LOCAL(local_types);
  *newtype = MPI_Type_c2f(local_new_type);

  free (c_indices);
}

void  mpi_type_struct__( count, blocklens, indices, old_types, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint * old_types;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_struct_( count, blocklens, indices, old_types, newtype, ierr );
}

void  MPI_TYPE_STRUCT( count, blocklens, indices, old_types, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint * old_types;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_struct_( count, blocklens, indices, old_types, newtype, ierr );
}

void  MPI_TYPE_STRUCT_( count, blocklens, indices, old_types, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint * old_types;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_struct_( count, blocklens, indices, old_types, newtype, ierr );
}

void  mpi_type_struct( count, blocklens, indices, old_types, newtype, ierr )
MPI_Fint *count;
MPI_Fint * blocklens;
MPI_Fint * indices;
MPI_Fint * old_types;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_struct_( count, blocklens, indices, old_types, newtype, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_type_ub_( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint *displacement;
MPI_Fint *ierr;
{
  MPI_Aint c_displacement;
  *ierr = MPI_Type_ub( MPI_Type_f2c(*datatype), &c_displacement );
  *displacement = c_displacement;
}

void   mpi_type_ub__( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint * displacement;
MPI_Fint *ierr;
{
  mpi_type_ub_( datatype, displacement, ierr );
}

void   MPI_TYPE_UB( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint * displacement;
MPI_Fint *ierr;
{
  mpi_type_ub_( datatype, displacement, ierr );
}

void   MPI_TYPE_UB_( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint * displacement;
MPI_Fint *ierr;
{
  mpi_type_ub_( datatype, displacement, ierr );
}


void   mpi_type_ub( datatype, displacement, ierr )
MPI_Fint *datatype;
MPI_Fint * displacement;
MPI_Fint *ierr;
{
  mpi_type_ub_( datatype, displacement, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_type_vector_( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  MPI_Datatype local_new_type;
  *ierr = MPI_Type_vector( *count, *blocklen, *stride, MPI_Type_f2c(*old_type), &local_new_type );
  *newtype = MPI_Type_c2f(local_new_type);
}

void  MPI_TYPE_VECTOR( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_vector_( count, blocklen, stride, old_type, newtype, ierr );
}

void  MPI_TYPE_VECTOR_( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_vector_( count, blocklen, stride, old_type, newtype, ierr );
}


void  mpi_type_vector( count, blocklen, stride, old_type, newtype, ierr )
MPI_Fint *count;
MPI_Fint *blocklen;
MPI_Fint *stride;
MPI_Fint *old_type;
MPI_Fint * newtype;
MPI_Fint *ierr;
{
  mpi_type_vector_( count, blocklen, stride, old_type, newtype, ierr );
} 

/******************************************************/
/******************************************************/

void   mpi_unpack_( inbuf, insize, position, outbuf, outcount, type, comm, ierr )
void * inbuf;
MPI_Fint *insize;
MPI_Fint * position;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint *type;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  *ierr = MPI_Unpack( inbuf, *insize, position, outbuf, *outcount, MPI_Type_f2c(*type), MPI_Comm_f2c(*comm) );
}

void   mpi_unpack__( inbuf, insize, position, outbuf, outcount, type, comm, ierr )
void * inbuf;
MPI_Fint *insize;
MPI_Fint * position;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint *type;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_unpack_( inbuf, insize, position, outbuf, outcount, type, comm, ierr );
}

void   MPI_UNPACK( inbuf, insize, position, outbuf, outcount, type, comm, ierr )
void * inbuf;
MPI_Fint *insize;
MPI_Fint * position;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint *type;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_unpack_( inbuf, insize, position, outbuf, outcount, type, comm, ierr );
}

void   MPI_UNPACK_( inbuf, insize, position, outbuf, outcount, type, comm, ierr )void * inbuf;
MPI_Fint *insize;
MPI_Fint * position;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint *type;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_unpack_( inbuf, insize, position, outbuf, outcount, type, comm, ierr );
}

void   mpi_unpack( inbuf, insize, position, outbuf, outcount, type, comm, ierr )
void * inbuf;
MPI_Fint *insize;
MPI_Fint * position;
void * outbuf;
MPI_Fint *outcount;
MPI_Fint *type;
MPI_Fint *comm;
MPI_Fint *ierr;
{
  mpi_unpack_( inbuf, insize, position, outbuf, outcount, type, comm, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_wait_( request, status, ierr )
MPI_Fint * request;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Request local_request = MPI_Request_f2c(*request);
  MPI_Status local_status;
  *ierr = MPI_Wait( &local_request, &local_status );
  *request = TAU_MPI_Request_c2f(local_request);
  MPI_Status_c2f(&local_status, status);
}

void   mpi_wait__( request, status, ierr )
MPI_Fint * request;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_wait_( request, status, ierr );
}

void   MPI_WAIT( request, status, ierr )
MPI_Fint * request;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_wait_( request, status, ierr );
}

void   MPI_WAIT_( request, status, ierr )
MPI_Fint * request;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_wait_( request, status, ierr );
}


void   mpi_wait( request, status, ierr )
MPI_Fint * request;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_wait_( request, status, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_waitall_( count, array_of_requests, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  TAU_DECL_LOCAL(MPI_Status, local_statuses);
  TAU_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *count);
  TAU_ALLOC_LOCAL(MPI_Status, local_statuses, *count);
  TAU_ASSIGN_VALUES(local_requests, array_of_requests, *count, MPI_Request_f2c);
  TAU_ASSIGN_STATUS_F2C(local_statuses, array_of_statuses, *count, MPI_Status_f2c);
  *ierr = MPI_Waitall( *count, local_requests, local_statuses );
  TAU_ASSIGN_VALUES(array_of_requests, local_requests, *count, TAU_MPI_Request_c2f);
  TAU_ASSIGN_STATUS_C2F(array_of_statuses, local_statuses, *count, MPI_Status_c2f);
  TAU_FREE_LOCAL(local_requests);
  TAU_FREE_LOCAL(local_statuses);
}

void  mpi_waitall__( count, array_of_requests, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_waitall_( count, array_of_requests, array_of_statuses, ierr );
}

void  MPI_WAITALL( count, array_of_requests, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_waitall_( count, array_of_requests, array_of_statuses, ierr );
}

void  MPI_WAITALL_( count, array_of_requests, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_waitall_( count, array_of_requests, array_of_statuses, ierr );
}

void  mpi_waitall( count, array_of_requests, array_of_statuses, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_waitall_( count, array_of_requests, array_of_statuses, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_waitany_( count, array_of_requests, index, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * status;
MPI_Fint *ierr;
{
  MPI_Status local_status;
  TAU_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *count);
  TAU_ASSIGN_VALUES(local_requests, array_of_requests, *count, MPI_Request_f2c);
  *ierr = MPI_Waitany( *count, local_requests, index, &local_status );
  TAU_ASSIGN_VALUES(array_of_requests, local_requests, *count, TAU_MPI_Request_c2f);
  MPI_Status_c2f(&local_status, status);
  TAU_FREE_LOCAL(local_requests);
  /* Increment the C index before returning it as a Fortran index as
     [0..N-1] => [1..N] array indexing differs in C and Fortran */
  (*index)++;
}

void  mpi_waitany__( count, array_of_requests, index, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_waitany_( count, array_of_requests, index, status, ierr );
}

void  MPI_WAITANY( count, array_of_requests, index, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * status; 
MPI_Fint *ierr;
{
  mpi_waitany_( count, array_of_requests, index, status, ierr );
}

void  MPI_WAITANY_( count, array_of_requests, index, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * status;
MPI_Fint *ierr;
{
  mpi_waitany_( count, array_of_requests, index, status, ierr );
}

void  mpi_waitany( count, array_of_requests, index, status, ierr )
MPI_Fint *count;
MPI_Fint * array_of_requests;
MPI_Fint * index;
MPI_Fint * status; 
MPI_Fint *ierr;
{
  mpi_waitany_( count, array_of_requests, index, status, ierr );
}


/******************************************************/
/******************************************************/

void  mpi_waitsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )

MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  int i;
  TAU_DECL_LOCAL(MPI_Status, local_statuses);
  TAU_DECL_ALLOC_LOCAL(MPI_Request, local_requests, *incount);
  TAU_ALLOC_LOCAL(MPI_Status, local_statuses, *incount);
  TAU_ASSIGN_VALUES(local_requests, array_of_requests, *incount, MPI_Request_f2c);
  TAU_ASSIGN_STATUS_F2C(local_statuses, array_of_statuses, *incount, MPI_Status_f2c);
  *ierr = MPI_Waitsome( *incount, local_requests, outcount, array_of_indices, local_statuses );
  TAU_ASSIGN_VALUES(array_of_requests, local_requests, *outcount, TAU_MPI_Request_c2f);
  TAU_ASSIGN_STATUS_C2F(array_of_statuses, local_statuses, *outcount, MPI_Status_c2f);
  TAU_FREE_LOCAL(local_requests);
  TAU_FREE_LOCAL(local_statuses);
  /* Increment the C index before returning it as a Fortran index as
     [0..N-1] => [1..N] array indexing differs in C and Fortran */
  for (i=0; i < *outcount; i++) array_of_indices[i]++;
}

void  mpi_waitsome__( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )

MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_waitsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr );
}

void  MPI_WAITSOME( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )

MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_waitsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr );
} 

void  MPI_WAITSOME_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )

MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_waitsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr );
}

void  mpi_waitsome( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr )

MPI_Fint *incount;
MPI_Fint * array_of_requests;
MPI_Fint * outcount;
MPI_Fint * array_of_indices;
MPI_Fint * array_of_statuses;
MPI_Fint *ierr;
{
  mpi_waitsome_( incount, array_of_requests, outcount, array_of_indices, array_of_statuses, ierr );
} 

/******************************************************/
/******************************************************/

void   mpi_cart_coords_( comm, rank, maxdims, coords, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint *maxdims;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  *ierr = MPI_Cart_coords( MPI_Comm_f2c(*comm), *rank, *maxdims, coords );
}

void   mpi_cart_coords__( comm, rank, maxdims, coords, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint *maxdims;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  mpi_cart_coords_( comm, rank, maxdims, coords, ierr );
}

void   MPI_CART_COORDS( comm, rank, maxdims, coords, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint *maxdims;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  mpi_cart_coords_( comm, rank, maxdims, coords, ierr );
}

void   MPI_CART_COORDS_( comm, rank, maxdims, coords, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint *maxdims;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  mpi_cart_coords_( comm, rank, maxdims, coords, ierr );
}

void   mpi_cart_coords( comm, rank, maxdims, coords, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint *maxdims;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  mpi_cart_coords_( comm, rank, maxdims, coords, ierr );
}

#ifdef TAU_MPI_CART_CREATE
/******************************************************/
/******************************************************/

#ifdef TAU_MPI_CART_CREATE
void   mpi_cart_create_( comm_old, ndims, dims, periods, reorder, comm_cart, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint *reorder;
MPI_Fint * comm_cart;
MPI_Fint *ierr;
{
  MPI_Comm local_comm_cart;

  *ierr = MPI_Cart_create( MPI_Comm_f2c(*comm_old), *ndims, dims, periods, *reorder, &local_comm_cart );
  *comm_cart = MPI_Comm_c2f(local_comm_cart);
}

void   mpi_cart_create__( comm_old, ndims, dims, periods, reorder, comm_cart, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint *reorder;
MPI_Fint * comm_cart;
MPI_Fint *ierr;
{
  mpi_cart_create_( comm_old, ndims, dims, periods, reorder, comm_cart, ierr );
}

void   MPI_CART_CREATE( comm_old, ndims, dims, periods, reorder, comm_cart, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint *reorder;
MPI_Fint * comm_cart;
MPI_Fint *ierr;
{
  mpi_cart_create_( comm_old, ndims, dims, periods, reorder, comm_cart, ierr );
}

void   MPI_CART_CREATE_( comm_old, ndims, dims, periods, reorder, comm_cart, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint *reorder;
MPI_Fint * comm_cart;
MPI_Fint *ierr;
{
  mpi_cart_create_( comm_old, ndims, dims, periods, reorder, comm_cart, ierr );
}


void   mpi_cart_create( comm_old, ndims, dims, periods, reorder, comm_cart, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint *reorder;
MPI_Fint * comm_cart;
MPI_Fint *ierr;
{
  mpi_cart_create_( comm_old, ndims, dims, periods, reorder, comm_cart, ierr );
}
#endif /* TAU_MPI_CART_CREATE */

#endif /* TAU_MPI_CART_CREATE */
/******************************************************/
/******************************************************/

void   mpi_cart_get_( comm, maxdims, dims, periods, coords, ierr )
MPI_Fint *comm;
MPI_Fint *maxdims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  *ierr = MPI_Cart_get( MPI_Comm_f2c(*comm), *maxdims, dims, periods, coords );
}

void   mpi_cart_get__( comm, maxdims, dims, periods, coords, ierr )
MPI_Fint *comm;
MPI_Fint *maxdims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  mpi_cart_get_( comm, maxdims, dims, periods, coords, ierr );
}

void   MPI_CART_GET( comm, maxdims, dims, periods, coords, ierr )
MPI_Fint *comm;
MPI_Fint *maxdims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  mpi_cart_get_( comm, maxdims, dims, periods, coords, ierr );
}

void   MPI_CART_GET_( comm, maxdims, dims, periods, coords, ierr )
MPI_Fint *comm;
MPI_Fint *maxdims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  mpi_cart_get_( comm, maxdims, dims, periods, coords, ierr );
}


void   mpi_cart_get( comm, maxdims, dims, periods, coords, ierr )
MPI_Fint *comm;
MPI_Fint *maxdims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * coords;
MPI_Fint *ierr;
{
  mpi_cart_get_( comm, maxdims, dims, periods, coords, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_cart_map_( comm_old, ndims, dims, periods, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  *ierr = MPI_Cart_map( MPI_Comm_f2c(*comm_old), *ndims, dims, periods, newrank );
}

void   mpi_cart_map__( comm_old, ndims, dims, periods, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  mpi_cart_map_( comm_old, ndims, dims, periods, newrank, ierr );
}

void   MPI_CART_MAP( comm_old, ndims, dims, periods, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  mpi_cart_map_( comm_old, ndims, dims, periods, newrank, ierr );
}

void   MPI_CART_MAP_( comm_old, ndims, dims, periods, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  mpi_cart_map_( comm_old, ndims, dims, periods, newrank, ierr );
}


void   mpi_cart_map( comm_old, ndims, dims, periods, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint * periods;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  mpi_cart_map_( comm_old, ndims, dims, periods, newrank, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_cart_rank_( comm, coords, rank, ierr )
MPI_Fint *comm;
MPI_Fint * coords;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  *ierr = MPI_Cart_rank( MPI_Comm_f2c(*comm), coords, rank );
}

void   mpi_cart_rank__( comm, coords, rank, ierr )
MPI_Fint *comm;
MPI_Fint * coords;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_cart_rank_( comm, coords, rank, ierr );
}

void   MPI_CART_RANK( comm, coords, rank, ierr )
MPI_Fint *comm;
MPI_Fint * coords;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_cart_rank_( comm, coords, rank, ierr );
}

void   MPI_CART_RANK_( comm, coords, rank, ierr )
MPI_Fint *comm;
MPI_Fint * coords;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_cart_rank_( comm, coords, rank, ierr );
}


void   mpi_cart_rank( comm, coords, rank, ierr )
MPI_Fint *comm;
MPI_Fint * coords;
MPI_Fint * rank;
MPI_Fint *ierr;
{
  mpi_cart_rank_( comm, coords, rank, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_cart_shift_( comm, direction, displ, source, dest, ierr )
MPI_Fint *comm;
MPI_Fint *direction;
MPI_Fint *displ;
MPI_Fint * source;
MPI_Fint * dest;
MPI_Fint *ierr;
{
  *ierr = MPI_Cart_shift( MPI_Comm_f2c(*comm), *direction, *displ, source, dest );
}

void   mpi_cart_shift__( comm, direction, displ, source, dest, ierr )
MPI_Fint *comm;
MPI_Fint *direction;
MPI_Fint *displ;
MPI_Fint * source;
MPI_Fint * dest;
MPI_Fint *ierr;
{
  mpi_cart_shift_( comm, direction, displ, source, dest, ierr );
}

void   MPI_CART_SHIFT( comm, direction, displ, source, dest, ierr )
MPI_Fint *comm;
MPI_Fint *direction;
MPI_Fint *displ;
MPI_Fint * source;
MPI_Fint * dest;
MPI_Fint *ierr;
{
  mpi_cart_shift_( comm, direction, displ, source, dest, ierr );
}

void   MPI_CART_SHIFT_( comm, direction, displ, source, dest, ierr )
MPI_Fint *comm;
MPI_Fint *direction;
MPI_Fint *displ;
MPI_Fint * source;
MPI_Fint * dest;
MPI_Fint *ierr;
{
  mpi_cart_shift_( comm, direction, displ, source, dest, ierr );
}


void   mpi_cart_shift( comm, direction, displ, source, dest, ierr )
MPI_Fint *comm;
MPI_Fint *direction;
MPI_Fint *displ;
MPI_Fint * source;
MPI_Fint * dest;
MPI_Fint *ierr;
{
  mpi_cart_shift_( comm, direction, displ, source, dest, ierr );
}


/******************************************************/
/******************************************************/
void   mpi_cart_sub_( comm, remain_dims, comm_new, ierr )
MPI_Fint *comm;
MPI_Fint * remain_dims;
MPI_Fint * comm_new;
MPI_Fint *ierr;
{
  MPI_Comm local_comm_new;
  *ierr = MPI_Cart_sub( MPI_Comm_f2c(*comm), remain_dims, &local_comm_new );
  *comm_new = MPI_Comm_c2f(local_comm_new);
}

void   mpi_cart_sub__( comm, remain_dims, comm_new, ierr )
MPI_Fint *comm;
MPI_Fint * remain_dims;
MPI_Fint * comm_new;
MPI_Fint *ierr;
{
  mpi_cart_sub_( comm, remain_dims, comm_new, ierr );
}

void   MPI_CART_SUB( comm, remain_dims, comm_new, ierr )
MPI_Fint *comm;
MPI_Fint * remain_dims;
MPI_Fint * comm_new;
MPI_Fint *ierr;
{
  mpi_cart_sub_( comm, remain_dims, comm_new, ierr );
}

void   MPI_CART_SUB_( comm, remain_dims, comm_new, ierr )
MPI_Fint *comm;
MPI_Fint * remain_dims;
MPI_Fint * comm_new;
MPI_Fint *ierr;
{
  mpi_cart_sub_( comm, remain_dims, comm_new, ierr );
}

void   mpi_cart_sub( comm, remain_dims, comm_new, ierr )
MPI_Fint *comm;
MPI_Fint * remain_dims;
MPI_Fint * comm_new;
MPI_Fint *ierr;
{
  mpi_cart_sub_( comm, remain_dims, comm_new, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_cartdim_get_( comm, ndims, ierr )
MPI_Fint *comm;
MPI_Fint * ndims;
MPI_Fint *ierr;
{
  *ierr = MPI_Cartdim_get( MPI_Comm_f2c(*comm), ndims );
}

void   mpi_cartdim_get__( comm, ndims, ierr )
MPI_Fint *comm;
MPI_Fint * ndims;
MPI_Fint *ierr;
{
  mpi_cartdim_get_( comm, ndims, ierr );
}

void   MPI_CARTDIM_GET( comm, ndims, ierr )
MPI_Fint *comm;
MPI_Fint * ndims;
MPI_Fint *ierr;
{
  mpi_cartdim_get_( comm, ndims, ierr );
}

void   MPI_CARTDIM_GET_( comm, ndims, ierr )
MPI_Fint *comm;
MPI_Fint * ndims;
MPI_Fint *ierr;
{
  mpi_cartdim_get_( comm, ndims, ierr );
}


void   mpi_cartdim_get( comm, ndims, ierr )
MPI_Fint *comm;
MPI_Fint * ndims;
MPI_Fint *ierr;
{
  mpi_cartdim_get_( comm, ndims, ierr );
}

/******************************************************/
/******************************************************/

void  mpi_dims_create_( nnodes, ndims, dims, ierr )
MPI_Fint *nnodes;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint *ierr;
{
  *ierr = MPI_Dims_create( *nnodes, *ndims, dims );
}

void  mpi_dims_create__( nnodes, ndims, dims, ierr )
MPI_Fint *nnodes;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint *ierr;
{
  mpi_dims_create_( nnodes, ndims, dims, ierr );
}

void  MPI_DIMS_CREATE( nnodes, ndims, dims, ierr )
MPI_Fint *nnodes;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint *ierr;
{
  mpi_dims_create_( nnodes, ndims, dims, ierr );
}

void  MPI_DIMS_CREATE_( nnodes, ndims, dims, ierr )
MPI_Fint *nnodes;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint *ierr;
{
  mpi_dims_create_( nnodes, ndims, dims, ierr );
}


void  mpi_dims_create( nnodes, ndims, dims, ierr )
MPI_Fint *nnodes;
MPI_Fint *ndims;
MPI_Fint * dims;
MPI_Fint *ierr;
{
  mpi_dims_create_( nnodes, ndims, dims, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_graph_create_( comm_old, nnodes, index, edges, reorder, comm_graph, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *reorder;
MPI_Fint * comm_graph;
MPI_Fint *ierr;
{
  MPI_Comm local_comm_graph;
  *ierr = MPI_Graph_create( MPI_Comm_f2c(*comm_old), *nnodes, index, edges, *reorder, &local_comm_graph );
  *comm_graph = MPI_Comm_c2f(local_comm_graph);
}

void   mpi_graph_create__( comm_old, nnodes, index, edges, reorder, comm_graph, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *reorder;
MPI_Fint * comm_graph;
MPI_Fint *ierr;
{
  mpi_graph_create_( comm_old, nnodes, index, edges, reorder, comm_graph, ierr );
}

void   MPI_GRAPH_CREATE( comm_old, nnodes, index, edges, reorder, comm_graph, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *reorder;
MPI_Fint * comm_graph;
MPI_Fint *ierr;
{
  mpi_graph_create_( comm_old, nnodes, index, edges, reorder, comm_graph, ierr );
} 

void   MPI_GRAPH_CREATE_( comm_old, nnodes, index, edges, reorder, comm_graph, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *reorder;
MPI_Fint * comm_graph;
MPI_Fint *ierr;
{
  mpi_graph_create_( comm_old, nnodes, index, edges, reorder, comm_graph, ierr );
}

void   mpi_graph_create( comm_old, nnodes, index, edges, reorder, comm_graph, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *reorder;
MPI_Fint * comm_graph;
MPI_Fint *ierr;
{
  mpi_graph_create_( comm_old, nnodes, index, edges, reorder, comm_graph, ierr );
} 

/******************************************************/
/******************************************************/

void   mpi_graph_get_( comm, maxindex, maxedges, index, edges, ierr )
MPI_Fint *comm;
MPI_Fint *maxindex;
MPI_Fint *maxedges;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *ierr;
{
  *ierr = MPI_Graph_get( MPI_Comm_f2c(*comm), *maxindex, *maxedges, index, edges );
}

void   mpi_graph_get__( comm, maxindex, maxedges, index, edges, ierr )
MPI_Fint *comm;
MPI_Fint *maxindex;
MPI_Fint *maxedges;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *ierr;
{
  mpi_graph_get_( comm, maxindex, maxedges, index, edges, ierr );
}

void   MPI_GRAPH_GET( comm, maxindex, maxedges, index, edges, ierr )
MPI_Fint *comm;
MPI_Fint *maxindex;
MPI_Fint *maxedges;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *ierr;
{
  mpi_graph_get_( comm, maxindex, maxedges, index, edges, ierr );
}

void   MPI_GRAPH_GET_( comm, maxindex, maxedges, index, edges, ierr )
MPI_Fint *comm;
MPI_Fint *maxindex;
MPI_Fint *maxedges;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *ierr;
{
  mpi_graph_get_( comm, maxindex, maxedges, index, edges, ierr );
}

void   mpi_graph_get( comm, maxindex, maxedges, index, edges, ierr )
MPI_Fint *comm;
MPI_Fint *maxindex;
MPI_Fint *maxedges;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint *ierr;
{
  mpi_graph_get_( comm, maxindex, maxedges, index, edges, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_graph_map_( comm_old, nnodes, index, edges, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  *ierr = MPI_Graph_map( MPI_Comm_f2c(*comm_old), *nnodes, index, edges, newrank );
}

void   mpi_graph_map__( comm_old, nnodes, index, edges, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  mpi_graph_map_( comm_old, nnodes, index, edges, newrank, ierr );
}

void   MPI_GRAPH_MAP( comm_old, nnodes, index, edges, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  mpi_graph_map_( comm_old, nnodes, index, edges, newrank, ierr );
}

void   MPI_GRAPH_MAP_( comm_old, nnodes, index, edges, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  mpi_graph_map_( comm_old, nnodes, index, edges, newrank, ierr );
}


void   mpi_graph_map( comm_old, nnodes, index, edges, newrank, ierr )
MPI_Fint *comm_old;
MPI_Fint *nnodes;
MPI_Fint * index;
MPI_Fint * edges;
MPI_Fint * newrank;
MPI_Fint *ierr;
{
  mpi_graph_map_( comm_old, nnodes, index, edges, newrank, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_graph_neighbors_( comm, rank, maxneighbors, neighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
int  *maxneighbors;
MPI_Fint * neighbors;
MPI_Fint *ierr;
{
  *ierr = MPI_Graph_neighbors( MPI_Comm_f2c(*comm), *rank, *maxneighbors, neighbors );
}

void   mpi_graph_neighbors__( comm, rank, maxneighbors, neighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
int  *maxneighbors;
MPI_Fint * neighbors;
MPI_Fint *ierr;
{
  mpi_graph_neighbors_( comm, rank, maxneighbors, neighbors, ierr );
}

void   MPI_GRAPH_NEIGHBORS( comm, rank, maxneighbors, neighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
int  *maxneighbors;
MPI_Fint * neighbors;
MPI_Fint *ierr;
{
  mpi_graph_neighbors_( comm, rank, maxneighbors, neighbors, ierr );
}

void   MPI_GRAPH_NEIGHBORS_( comm, rank, maxneighbors, neighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
int  *maxneighbors;
MPI_Fint * neighbors;
MPI_Fint *ierr;
{
  mpi_graph_neighbors_( comm, rank, maxneighbors, neighbors, ierr );
}

void   mpi_graph_neighbors( comm, rank, maxneighbors, neighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
int  *maxneighbors;
MPI_Fint * neighbors;
MPI_Fint *ierr;
{
  mpi_graph_neighbors_( comm, rank, maxneighbors, neighbors, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_graph_neighbors_count_( comm, rank, nneighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint * nneighbors;
MPI_Fint *ierr;
{
  *ierr = MPI_Graph_neighbors_count( MPI_Comm_f2c(*comm), *rank, nneighbors );
}

void   mpi_graph_neighbors_count__( comm, rank, nneighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint * nneighbors;
MPI_Fint *ierr;
{
  mpi_graph_neighbors_count_( comm, rank, nneighbors, ierr );
}

void   MPI_GRAPH_NEIGHBORS_COUNT( comm, rank, nneighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint * nneighbors;
MPI_Fint *ierr;
{
  mpi_graph_neighbors_count_( comm, rank, nneighbors, ierr );
}

void   MPI_GRAPH_NEIGHBORS_COUNT_( comm, rank, nneighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint * nneighbors;
MPI_Fint *ierr;
{
  mpi_graph_neighbors_count_( comm, rank, nneighbors, ierr );
}

void   mpi_graph_neighbors_count( comm, rank, nneighbors, ierr )
MPI_Fint *comm;
MPI_Fint *rank;
MPI_Fint * nneighbors;
MPI_Fint *ierr;
{
  mpi_graph_neighbors_count_( comm, rank, nneighbors, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_graphdims_get_( comm, nnodes, nedges, ierr )
MPI_Fint *comm;
MPI_Fint * nnodes;
MPI_Fint * nedges;
MPI_Fint *ierr;
{
  *ierr = MPI_Graphdims_get( MPI_Comm_f2c(*comm), nnodes, nedges );
}

void   mpi_graphdims_get__( comm, nnodes, nedges, ierr )
MPI_Fint *comm;
MPI_Fint * nnodes;
MPI_Fint * nedges;
MPI_Fint *ierr;
{
  mpi_graphdims_get_( comm, nnodes, nedges, ierr );
}

void   MPI_GRAPHDIMS_GET( comm, nnodes, nedges, ierr )
MPI_Fint *comm;
MPI_Fint * nnodes;
MPI_Fint * nedges;
MPI_Fint *ierr;
{
  mpi_graphdims_get_( comm, nnodes, nedges, ierr );
}

void   MPI_GRAPHDIMS_GET_( comm, nnodes, nedges, ierr )
MPI_Fint *comm;
MPI_Fint * nnodes;
MPI_Fint * nedges;
MPI_Fint *ierr;
{
  mpi_graphdims_get_( comm, nnodes, nedges, ierr );
}

void   mpi_graphdims_get( comm, nnodes, nedges, ierr )
MPI_Fint *comm;
MPI_Fint * nnodes;
MPI_Fint * nedges;
MPI_Fint *ierr;
{
  mpi_graphdims_get_( comm, nnodes, nedges, ierr );
}

/******************************************************/
/******************************************************/

void   mpi_topo_test_( comm, top_type, ierr )
MPI_Fint *comm;
MPI_Fint * top_type;
MPI_Fint *ierr;
{
  *ierr = MPI_Topo_test( MPI_Comm_f2c(*comm), top_type );
}

void   mpi_topo_test__( comm, top_type, ierr )
MPI_Fint *comm;
MPI_Fint * top_type;
MPI_Fint *ierr;
{
  mpi_topo_test_( comm, top_type, ierr );
}

void   MPI_TOPO_TEST( comm, top_type, ierr )
MPI_Fint *comm;
MPI_Fint * top_type;
MPI_Fint *ierr;
{
  mpi_topo_test_( comm, top_type, ierr );
}

void   MPI_TOPO_TEST_( comm, top_type, ierr )
MPI_Fint *comm;
MPI_Fint * top_type;
MPI_Fint *ierr;
{
  mpi_topo_test_( comm, top_type, ierr );
}


void   mpi_topo_test( comm, top_type, ierr )
MPI_Fint *comm;
MPI_Fint * top_type;
MPI_Fint *ierr;
{
  mpi_topo_test_( comm, top_type, ierr );
}

/******************************************************/
/******************************************************/
