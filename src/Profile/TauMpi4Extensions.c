#include <mpi.h>
#include <TAU.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <Profile/TauUtil.h>
#include <Profile/TauEnv.h>
#include "check_mpi_version.h"

int MPI_Isendrecv(const void* sendbuf, int sendcount, MPI_Datatype
    sendtype, int dest, int sendtag, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Isendrecv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Isendrecv( sendbuf, sendcount, sendtype, dest, sendtag,
                              recvbuf, recvcount, recvtype, source, recvtag,
                              comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Isendrecv_replace(void* buf, int count, MPI_Datatype
    datatype, int dest, int sendtag, int source, int recvtag,
    MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Isendrecv_replace()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Isendrecv_replace( buf, count, datatype, dest, sendtag,
                              source, recvtag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Comm_idup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm*
    newcomm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_idup_with_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_idup_with_info( comm, info, newcomm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Info_get_string(MPI_Info info, const char* key, int* buflen,
    char* value, int* flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_get_string()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_get_string( info, key, buflen, value, flag ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Info_create_env(int argc, char **argv, MPI_Info* info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_create_env()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_create_env( argc, argv, info ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}


//Large count functions and _init

int MPI_Accumulate_c(const void* origin_addr, MPI_Count
    origin_count, MPI_Datatype origin_datatype, int target_rank,
    MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype
    target_datatype, MPI_Op op, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Accumulate_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Accumulate_c( origin_addr, origin_count, origin_datatype,
                                target_rank, target_disp, target_count, 
                                target_datatype, op, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Raccumulate_c(const void* origin_addr, MPI_Count
    origin_count, MPI_Datatype origin_datatype, int target_rank,
    MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype
    target_datatype, MPI_Op op, MPI_Win win, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Raccumulate_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Raccumulate_c( origin_addr, origin_count, origin_datatype,
                                target_rank, target_disp, target_count, 
                                target_datatype, op, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}


int MPI_Allgather_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allgather_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allgather_c( sendbuf, sendcount, sendtype, 
                                recvbuf, recvcount, recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iallgather_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iallgather_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iallgather_c( sendbuf, sendcount, sendtype, 
                                recvbuf, recvcount, recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Allgather_init(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allgather_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allgather_init( sendbuf, sendcount, sendtype, 
                                recvbuf, recvcount, recvtype, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Allgather_init_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allgather_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allgather_init_c( sendbuf, sendcount, sendtype, 
                                recvbuf, recvcount, recvtype, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Allgatherv_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allgatherv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allgatherv_c( sendbuf, sendcount, sendtype, 
                                recvbuf, recvcounts, displs,
                                recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iallgatherv_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iallgatherv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iallgatherv_c( sendbuf, sendcount, sendtype, 
                                recvbuf, recvcounts, displs,
                                recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Allgatherv_init(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
    const int displs[], MPI_Datatype recvtype, MPI_Comm comm,
    MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allgatherv_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allgatherv_init( sendbuf, sendcount, sendtype, 
                                recvbuf, recvcounts, displs,
                                recvtype, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Allgatherv_init_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allgatherv_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allgatherv_init_c( sendbuf, sendcount, sendtype, 
                                recvbuf, recvcounts, displs,
                                recvtype, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Allreduce_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allreduce_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allreduce_c( sendbuf, recvbuf, count, 
                                datatype, op, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iallreduce_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iallreduce_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iallreduce_c( sendbuf, recvbuf, count, 
                                datatype, op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Allreduce_init(const void* sendbuf, void* recvbuf, int
    count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Info
    info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allreduce_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allreduce_init( sendbuf, recvbuf, count, datatype,
                                op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Allreduce_init_c(const void* sendbuf, void* recvbuf,
    MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm
    comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Allreduce_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Allreduce_init_c( sendbuf, recvbuf, count, datatype,
                                op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoall_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoall_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoall_c( sendbuf, sendcount, sendtype,
                               recvbuf, recvcount, recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ialltoall_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ialltoall_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ialltoall_c( sendbuf, sendcount, sendtype,
                               recvbuf, recvcount, recvtype, 
                               comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoall_init(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoall_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoall_init( sendbuf, sendcount, sendtype,
                               recvbuf, recvcount, recvtype, 
                               comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoall_init_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoall_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoall_init_c( sendbuf, sendcount, sendtype,
                               recvbuf, recvcount, recvtype, 
                               comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoallv_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], MPI_Datatype sendtype,
    void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint
    rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoallv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoallv_c( sendbuf, sendcounts, sdispls, sendtype,
                               recvbuf, recvcounts, rdispls, recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ialltoallv_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], MPI_Datatype sendtype,
    void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint
    rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ialltoallv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ialltoallv_c( sendbuf, sendcounts, sdispls, sendtype,
                               recvbuf, recvcounts, rdispls, recvtype, 
                               comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoallv_init(const void* sendbuf, const int sendcounts[],
    const int sdispls[], MPI_Datatype sendtype, void* recvbuf, const
    int recvcounts[], const int rdispls[], MPI_Datatype recvtype,
    MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoallv_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoallv_init( sendbuf, sendcounts, sdispls, sendtype,
                               recvbuf, recvcounts, rdispls, recvtype, 
                               comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoallv_init_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], MPI_Datatype sendtype,
    void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint
    rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoallv_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoallv_init_c( sendbuf, sendcounts, sdispls, sendtype,
                               recvbuf, recvcounts, rdispls, recvtype, 
                               comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoallw_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype
    sendtypes[], void* recvbuf, const MPI_Count recvcounts[], const
    MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm
    comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoallw_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoallw_c( sendbuf, sendcounts, sdispls, sendtypes,
                               recvbuf, recvcounts, rdispls, recvtypes, 
                               comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ialltoallw_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype
    sendtypes[], void* recvbuf, const MPI_Count recvcounts[], const
    MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm
    comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ialltoallw_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ialltoallw_c( sendbuf, sendcounts, sdispls, sendtypes,
                               recvbuf, recvcounts, rdispls, recvtypes, 
                               comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoallw_init(const void* sendbuf, const int sendcounts[],
    const int sdispls[], const MPI_Datatype sendtypes[], void*
    recvbuf, const int recvcounts[], const int rdispls[], const
    MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoallw_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoallw_init( sendbuf, sendcounts, sdispls, sendtypes,
                               recvbuf, recvcounts, rdispls, recvtypes, 
                               comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Alltoallw_init_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype
    sendtypes[], void* recvbuf, const MPI_Count recvcounts[], const
    MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm
    comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoallw_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoallw_init_c( sendbuf, sendcounts, sdispls, sendtypes,
                               recvbuf, recvcounts, rdispls, recvtypes, 
                               comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Barrier_init(MPI_Comm comm, MPI_Info info, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Barrier_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Barrier_init( comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Bcast_c(void* buffer, MPI_Count count, MPI_Datatype
    datatype, int root, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Bcast_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Bcast_c( buffer, count, datatype, 
                            root, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ibcast_c(void* buffer, MPI_Count count, MPI_Datatype
    datatype, int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ibcast_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ibcast_c( buffer, count, datatype, 
                            root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Bcast_init(void* buffer, int count, MPI_Datatype datatype,
    int root, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Bcast_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Bcast_init( buffer, count, datatype, 
                            root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Bcast_init_c(void* buffer, MPI_Count count, MPI_Datatype
    datatype, int root, MPI_Comm comm, MPI_Info info, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Bcast_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Bcast_init_c( buffer, count, datatype, 
                            root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Bsend_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Bsend_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Bsend_c( buf, count, datatype, 
                            dest, tag, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Bsend_init_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Bsend_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Bsend_init_c( buf, count, datatype, 
                            dest, tag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Buffer_attach_c(void* buffer, MPI_Count size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Buffer_attach_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Buffer_attach_c( buffer, size ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Buffer_detach_c(void* buffer_addr, MPI_Count* size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Buffer_detach_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Buffer_detach_c( buffer_addr, size ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Exscan_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Exscan_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Exscan_c( sendbuf, recvbuf, count,
                            datatype, op, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iexscan_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iexscan_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iexscan_c( sendbuf, recvbuf, count,
                            datatype, op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Exscan_init(const void* sendbuf, void* recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Exscan_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Exscan_init( sendbuf, recvbuf, count,
                            datatype, op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Exscan_init_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Info
    info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Exscan_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Exscan_init_c( sendbuf, recvbuf, count,
                            datatype, op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iread_c(MPI_File fh, void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_c( fh, buf, count, 
                                datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iread_all_c(MPI_File fh, void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_all_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_all_c( fh, buf, count, 
                                datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iread_at_c(MPI_File fh, MPI_Offset offset, void* buf,
    MPI_Count count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_at_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_at_c( fh, offset, buf, count, 
                                datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}



int MPI_Comm_create_from_group(MPI_Group group, const char*
    stringtag, MPI_Info info, MPI_Errhandler errhandler, MPI_Comm*
    newcomm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_create_from_group()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_create_from_group( group, stringtag, info,
                                    errhandler, newcomm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Group_from_session_pset(MPI_Session session, const char*
    pset_name, MPI_Group* newgroup)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Group_from_session_pset()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Group_from_session_pset( session, pset_name, newgroup ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Intercomm_create_from_groups(MPI_Group local_group, int
    local_leader, MPI_Group remote_group, int remote_leader, const
    char* stringtag, MPI_Info info, MPI_Errhandler errhandler,
    MPI_Comm* newintercomm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Intercomm_create_from_groups()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Intercomm_create_from_groups( local_group, local_leader,
                                    remote_group, remote_leader, stringtag,
                                    info, errhandler, newintercomm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Isendrecv_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, int dest, int sendtag, void* recvbuf,
    MPI_Count recvcount, MPI_Datatype recvtype, int source, int
    recvtag, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Isendrecv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Isendrecv_c( sendbuf, sendcount, sendtype, dest, sendtag,
                              recvbuf, recvcount, recvtype, source, recvtag,
                              comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Isendrecv_replace_c(void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int sendtag, int source, int recvtag,
    MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Isendrecv_replace_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Isendrecv_replace_c( buf, count, datatype, dest, sendtag,
                              source, recvtag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}


int MPI_File_iread_at_all_c(MPI_File fh, MPI_Offset offset, void*
    buf, MPI_Count count, MPI_Datatype datatype, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_at_all_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_at_all_c( fh, offset, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iread_shared_c(MPI_File fh, void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_shared_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_shared_c( fh, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iwrite_c(MPI_File fh, const void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_c( fh, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iwrite_all_c(MPI_File fh, const void* buf, MPI_Count
    count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_all_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_all_c( fh, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iwrite_at_c(MPI_File fh, MPI_Offset offset, const void*
    buf, MPI_Count count, MPI_Datatype datatype, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_at_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_at_c( fh, offset, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iwrite_at_all_c(MPI_File fh, MPI_Offset offset, const
    void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_at_all_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_at_all_c( fh, offset, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iwrite_shared_c(MPI_File fh, const void* buf, MPI_Count
    count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_shared_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_shared_c( fh, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_c(MPI_File fh, void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_c( fh, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_all_c(MPI_File fh, void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_all_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_all_c( fh, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_all_begin_c(MPI_File fh, void* buf, MPI_Count
    count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_all_begin_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_all_begin_c( fh, buf, count, datatype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_at_c(MPI_File fh, MPI_Offset offset, void* buf,
    MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_at_c( fh, offset, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_at_all_c(MPI_File fh, MPI_Offset offset, void*
    buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_all_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_at_all_c( fh, offset, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_at_all_begin_c(MPI_File fh, MPI_Offset offset,
    void* buf, MPI_Count count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_all_begin_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_at_all_begin_c( fh, offset, buf, count, datatype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_ordered_c(MPI_File fh, void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_ordered_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_ordered_c( fh, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_ordered_begin_c(MPI_File fh, void* buf, MPI_Count
    count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_ordered_begin_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_ordered_begin_c( fh, buf, count, datatype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_read_shared_c(MPI_File fh, void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_shared_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_shared_c( fh, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_c(MPI_File fh, const void* buf, MPI_Count count,
    MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_c( fh, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_all_c(MPI_File fh, const void* buf, MPI_Count
    count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_all_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_all_c( fh, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_all_begin_c(MPI_File fh, const void* buf,
    MPI_Count count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_all_begin_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_all_begin_c( fh, buf, count, datatype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_at_c(MPI_File fh, MPI_Offset offset, const void*
    buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_at_c( fh, offset, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_at_all_c(MPI_File fh, MPI_Offset offset, const
    void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status*
    status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_all_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_at_all_c( fh, offset, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_at_all_begin_c(MPI_File fh, MPI_Offset offset,
    const void* buf, MPI_Count count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_all_begin_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_at_all_begin_c( fh, offset, buf, count, datatype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_ordered_c(MPI_File fh, const void* buf, MPI_Count
    count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_ordered_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_ordered_c( fh, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_ordered_begin_c(MPI_File fh, const void* buf,
    MPI_Count count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_ordered_begin_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_ordered_begin_c( fh, buf, count, datatype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_write_shared_c(MPI_File fh, const void* buf, MPI_Count
    count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_shared_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_shared_c( fh, buf, count, datatype, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Gather_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Gather_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Gather_c( sendbuf, sendcount, sendtype,
                            recvbuf, recvcount, recvtype, 
                            root, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Igather_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Igather_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Igather_c( sendbuf, sendcount, sendtype,
                            recvbuf, recvcount, recvtype, 
                            root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Gather_init(const void* sendbuf, int sendcount, MPI_Datatype
    sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Gather_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Gather_init( sendbuf, sendcount, sendtype,
                            recvbuf, recvcount, recvtype, 
                            root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Gather_init_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Gather_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Gather_init_c( sendbuf, sendcount, sendtype,
                            recvbuf, recvcount, recvtype, 
                            root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Gatherv_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    int root, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Gatherv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Gatherv_c( sendbuf, sendcount, sendtype,
                            recvbuf, recvcounts, displs,
                            recvtype, root, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Igatherv_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Igatherv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Igatherv_c( sendbuf, sendcount, sendtype,
                            recvbuf, recvcounts, displs,
                            recvtype, root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Gatherv_init(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
    const int displs[], MPI_Datatype recvtype, int root, MPI_Comm
    comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Gatherv_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Gatherv_init( sendbuf, sendcount, sendtype,
                            recvbuf, recvcounts, displs,
                            recvtype, root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Gatherv_init_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    int root, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Gatherv_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Gatherv_init_c( sendbuf, sendcount, sendtype,
                            recvbuf, recvcounts, displs,
                            recvtype, root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Get_c(void* origin_addr, MPI_Count origin_count,
    MPI_Datatype origin_datatype, int target_rank, MPI_Aint
    target_disp, MPI_Count target_count, MPI_Datatype
    target_datatype, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_c( origin_addr, origin_count, origin_datatype,
                         target_rank, target_disp, target_count,
                         target_datatype, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Rget_c(void* origin_addr, MPI_Count origin_count,
    MPI_Datatype origin_datatype, int target_rank, MPI_Aint
    target_disp, MPI_Count target_count, MPI_Datatype
    target_datatype, MPI_Win win, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Rget_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Rget_c( origin_addr, origin_count, origin_datatype,
                         target_rank, target_disp, target_count,
                         target_datatype, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Get_accumulate_c(const void* origin_addr, MPI_Count
    origin_count, MPI_Datatype origin_datatype, void* result_addr,
    MPI_Count result_count, MPI_Datatype result_datatype, int
    target_rank, MPI_Aint target_disp, MPI_Count target_count,
    MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_accumulate_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_accumulate_c( origin_addr, origin_count, origin_datatype,
                         result_addr, result_count, result_datatype,
                         target_rank, target_disp, target_count,
                         target_datatype, op, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Rget_accumulate_c(const void* origin_addr, MPI_Count
    origin_count, MPI_Datatype origin_datatype, void* result_addr,
    MPI_Count result_count, MPI_Datatype result_datatype, int
    target_rank, MPI_Aint target_disp, MPI_Count target_count,
    MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Rget_accumulate_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Rget_accumulate_c( origin_addr, origin_count, origin_datatype,
                         result_addr, result_count, result_datatype,
                         target_rank, target_disp, target_count,
                         target_datatype, op, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Get_count_c(const MPI_Status* status, MPI_Datatype datatype,
    MPI_Count* count)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_count_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_count_c( status, datatype, count ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

//New version of MPI_Get_elements_x
int MPI_Get_elements_c(const MPI_Status* status, MPI_Datatype
    datatype, MPI_Count* count)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_elements_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_elements_c( status, datatype, count ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ibsend_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ibsend_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ibsend_c( buf, count, datatype, dest, 
                            tag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Imrecv_c(void* buf, MPI_Count count, MPI_Datatype datatype,
    MPI_Message* message, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Imrecv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Imrecv_c( buf, count, datatype, 
                            message, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_allgather_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count
    recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgather_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_allgather_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ineighbor_allgather_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count
    recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_allgather_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_allgather_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_allgather_init(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgather_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_allgather_init( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_allgather_init_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count
    recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgather_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_allgather_init_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_allgatherv_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgatherv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_allgatherv_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcounts, displs, recvtype, 
                            comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ineighbor_allgatherv_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_allgatherv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_allgatherv_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcounts, displs, recvtype, 
                            comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_allgatherv_init(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
    const int displs[], MPI_Datatype recvtype, MPI_Comm comm,
    MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgatherv_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_allgatherv_init( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcounts, displs, recvtype, 
                            comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_allgatherv_init_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint displs[], MPI_Datatype recvtype,
    MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgatherv_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_allgatherv_init_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcounts, displs, recvtype, 
                            comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoall_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count
    recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoall_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoall_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype, 
                            comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ineighbor_alltoall_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count
    recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoall_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_alltoall_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype, 
                            comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoall_init(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoall_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoall_init( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype, 
                            comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoall_init_c(const void* sendbuf, MPI_Count
    sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count
    recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoall_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoall_init_c( sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype, 
                            comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoallv_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], MPI_Datatype sendtype,
    void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint
    rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoallv_c( sendbuf, sendcounts, sdispls, sendtype, 
                            recvbuf, recvcounts, rdispls, recvtype, 
                            comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ineighbor_alltoallv_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], MPI_Datatype sendtype,
    void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint
    rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoallv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_alltoallv_c( sendbuf, sendcounts, sdispls, sendtype, 
                            recvbuf, recvcounts, rdispls, recvtype, 
                            comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoallv_init(const void* sendbuf, const int
    sendcounts[], const int sdispls[], MPI_Datatype sendtype, void*
    recvbuf, const int recvcounts[], const int rdispls[],
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallv_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoallv_init( sendbuf, sendcounts, sdispls, sendtype, 
                            recvbuf, recvcounts, rdispls, recvtype, 
                            comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoallv_init_c(const void* sendbuf, const
    MPI_Count sendcounts[], const MPI_Aint sdispls[], MPI_Datatype
    sendtype, void* recvbuf, const MPI_Count recvcounts[], const
    MPI_Aint rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
    MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallv_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoallv_init_c( sendbuf, sendcounts, sdispls, sendtype, 
                            recvbuf, recvcounts, rdispls, recvtype, 
                            comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoallw_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype
    sendtypes[], void* recvbuf, const MPI_Count recvcounts[], const
    MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm
    comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallw_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoallw_c( sendbuf, sendcounts, sdispls, sendtypes, 
                            recvbuf, recvcounts, rdispls, recvtypes, 
                            comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ineighbor_alltoallw_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype
    sendtypes[], void* recvbuf, const MPI_Count recvcounts[], const
    MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm
    comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoallw_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_alltoallw_c( sendbuf, sendcounts, sdispls, sendtypes, 
                            recvbuf, recvcounts, rdispls, recvtypes, 
                            comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoallw_init(const void* sendbuf, const int
    sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype
    sendtypes[], void* recvbuf, const int recvcounts[], const
    MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm
    comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallw_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoallw_init( sendbuf, sendcounts, sdispls, sendtypes, 
                            recvbuf, recvcounts, rdispls, recvtypes, 
                            comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Neighbor_alltoallw_init_c(const void* sendbuf, const
    MPI_Count sendcounts[], const MPI_Aint sdispls[], const
    MPI_Datatype sendtypes[], void* recvbuf, const MPI_Count
    recvcounts[], const MPI_Aint rdispls[], const MPI_Datatype
    recvtypes[], MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallw_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoallw_init_c( sendbuf, sendcounts, sdispls, sendtypes, 
                            recvbuf, recvcounts, rdispls, recvtypes, 
                            comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Irecv_c(void* buf, MPI_Count count, MPI_Datatype datatype,
    int source, int tag, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Irecv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Irecv_c( buf, count, datatype, 
                           source, tag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm
    comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_c( sendbuf, recvbuf, count, datatype, 
                           op, root, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ireduce_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm
    comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ireduce_c( sendbuf, recvbuf, count, datatype, 
                           op, root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_init(const void* sendbuf, void* recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
    MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_init( sendbuf, recvbuf, count, datatype, 
                           op, root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_init_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm
    comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_init_c( sendbuf, recvbuf, count, datatype, 
                           op, root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_scatter_c(const void* sendbuf, void* recvbuf, const
    MPI_Count recvcounts[], MPI_Datatype datatype, MPI_Op op,
    MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_scatter_c( sendbuf, recvbuf, recvcounts, datatype, 
                           op, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ireduce_scatter_c(const void* sendbuf, void* recvbuf, const
    MPI_Count recvcounts[], MPI_Datatype datatype, MPI_Op op,
    MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_scatter_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ireduce_scatter_c( sendbuf, recvbuf, recvcounts, datatype, 
                           op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_scatter_init(const void* sendbuf, void* recvbuf,
    const int recvcounts[], MPI_Datatype datatype, MPI_Op op,
    MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_scatter_init( sendbuf, recvbuf, recvcounts, datatype, 
                           op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_scatter_init_c(const void* sendbuf, void* recvbuf,
    const MPI_Count recvcounts[], MPI_Datatype datatype, MPI_Op op,
    MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_scatter_init_c( sendbuf, recvbuf, recvcounts, datatype, 
                           op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_scatter_block_c(const void* sendbuf, void* recvbuf,
    MPI_Count recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm
    comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_block_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_scatter_block_c( sendbuf, recvbuf, recvcount, datatype, 
                           op, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ireduce_scatter_block_c(const void* sendbuf, void* recvbuf,
    MPI_Count recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm
    comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_scatter_block_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ireduce_scatter_block_c( sendbuf, recvbuf, recvcount, datatype, 
                           op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_scatter_block_init(const void* sendbuf, void*
    recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op,
    MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_block_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_scatter_block_init( sendbuf, recvbuf, recvcount, datatype, 
                           op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_scatter_block_init_c(const void* sendbuf, void*
    recvbuf, MPI_Count recvcount, MPI_Datatype datatype, MPI_Op op,
    MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_block_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_scatter_block_init_c( sendbuf, recvbuf, recvcount, datatype, 
                           op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Irsend_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Irsend_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Irsend_c( buf, count, datatype, 
                           dest, tag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scan_c(const void* sendbuf, void* recvbuf, MPI_Count count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scan_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scan_c( sendbuf, recvbuf, count, datatype, 
                           op, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Iscan_c(const void* sendbuf, void* recvbuf, MPI_Count count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iscan_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iscan_c( sendbuf, recvbuf, count, datatype, 
                           op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scan_init(const void* sendbuf, void* recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scan_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scan_init( sendbuf, recvbuf, count, datatype, 
                           op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scan_init_c(const void* sendbuf, void* recvbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Info
    info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scan_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scan_init_c( sendbuf, recvbuf, count, datatype, 
                           op, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scatter_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scatter_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scatter_c( sendbuf, sendcount, sendtype,
                           recvbuf, recvcount, recvtype, 
                           root, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Iscatter_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iscatter_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iscatter_c( sendbuf, sendcount, sendtype,
                           recvbuf, recvcount, recvtype, 
                           root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scatter_init(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scatter_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scatter_init( sendbuf, sendcount, sendtype,
                           recvbuf, recvcount, recvtype, 
                           root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scatter_init_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scatter_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scatter_init_c( sendbuf, sendcount, sendtype,
                           recvbuf, recvcount, recvtype, 
                           root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scatterv_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint displs[], MPI_Datatype sendtype,
    void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype, int
    root, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scatterv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scatterv_c( sendbuf, sendcounts, displs, sendtype,
                           recvbuf, recvcount, recvtype, 
                           root, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Iscatterv_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint displs[], MPI_Datatype sendtype,
    void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype, int
    root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iscatterv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iscatterv_c( sendbuf, sendcounts, displs, sendtype,
                           recvbuf, recvcount, recvtype, 
                           root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scatterv_init(const void* sendbuf, const int sendcounts[],
    const int displs[], MPI_Datatype sendtype, void* recvbuf, int
    recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
    MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scatterv_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scatterv_init( sendbuf, sendcounts, displs, sendtype,
                           recvbuf, recvcount, recvtype, 
                           root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Scatterv_init_c(const void* sendbuf, const MPI_Count
    sendcounts[], const MPI_Aint displs[], MPI_Datatype sendtype,
    void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype, int
    root, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Scatterv_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Scatterv_init_c( sendbuf, sendcounts, displs, sendtype,
                           recvbuf, recvcount, recvtype, 
                           root, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Isend_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Isend_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Isend_c( buf, count, datatype, dest, tag,
                           comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Issend_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Issend_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Issend_c( buf, count, datatype, dest, tag,
                           comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Mrecv_c(void* buf, MPI_Count count, MPI_Datatype datatype,
    MPI_Message* message, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Mrecv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Mrecv_c( buf, count, datatype,
                           message, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Op_create_c(MPI_User_function_c* user_fn, int commute,
    MPI_Op* op)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Op_create_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Op_create_c( user_fn, commute, op ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Pack_c(const void* inbuf, MPI_Count incount, MPI_Datatype
    datatype, void* outbuf, MPI_Count outsize, MPI_Count* position,
    MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pack_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pack_c( inbuf, incount, datatype, 
                          outbuf, outsize, position, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Pack_external_c(const char datarep[], const void* inbuf,
    MPI_Count incount, MPI_Datatype datatype, void* outbuf,
    MPI_Count outsize, MPI_Count* position)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pack_external_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pack_external_c( datarep, inbuf, incount, datatype,
                                  outbuf, outsize, position ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Pack_external_size_c(const char datarep[], MPI_Count
    incount, MPI_Datatype datatype, MPI_Count* size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pack_external_size_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pack_external_size_c( datarep, incount,
                                  datatype, size ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Pack_size_c(MPI_Count incount, MPI_Datatype datatype,
    MPI_Comm comm, MPI_Count* size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pack_size_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pack_size_c( incount, datatype, comm, size ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Parrived(MPI_Request request, int partition, int* flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Parrived()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Parrived( request, partition, flag ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Pready(int partition, MPI_Request request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pready()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pready( partition, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Pready_list(int length, const int array_of_partitions[],
    MPI_Request request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pready_list()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pready_list( length, array_of_partitions, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Pready_range(int partition_low, int partition_high,
    MPI_Request request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pready_range()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pready_range( partition_low, partition_high, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Precv_init(void* buf, int partitions, MPI_Count count,
    MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
    MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Precv_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Precv_init( buf, partitions, count, datatype,
                              source, tag, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Psend_init(const void* buf, int partitions, MPI_Count count,
    MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
    MPI_Info info, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Psend_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Psend_init( buf, partitions, count, datatype,
                              dest, tag, comm, info, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Put_c(const void* origin_addr, MPI_Count origin_count,
    MPI_Datatype origin_datatype, int target_rank, MPI_Aint
    target_disp, MPI_Count target_count, MPI_Datatype
    target_datatype, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Put_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Put_c( origin_addr, origin_count, origin_datatype,
                          target_rank, target_disp, target_count,
                          target_datatype, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Rput_c(const void* origin_addr, MPI_Count origin_count,
    MPI_Datatype origin_datatype, int target_rank, MPI_Aint
    target_disp, MPI_Count target_count, MPI_Datatype
    target_datatype, MPI_Win win, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Rput_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Rput_c( origin_addr, origin_count, origin_datatype,
                          target_rank, target_disp, target_count,
                          target_datatype, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Recv_c(void* buf, MPI_Count count, MPI_Datatype datatype,
    int source, int tag, MPI_Comm comm, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Recv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Recv_c( buf, count, datatype,
                          source, tag, comm, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Recv_init_c(void* buf, MPI_Count count, MPI_Datatype
    datatype, int source, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Recv_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Recv_init_c( buf, count, datatype,
                          source, tag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Reduce_local_c(const void* inbuf, void* inoutbuf, MPI_Count
    count, MPI_Datatype datatype, MPI_Op op)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_local_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_local_c( inbuf, inoutbuf, count,
                          datatype, op ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Register_datarep_c(const char* datarep,
    MPI_Datarep_conversion_function_c* read_conversion_fn,
    MPI_Datarep_conversion_function_c* write_conversion_fn,
    MPI_Datarep_extent_function* dtype_file_extent_fn, void*
    extra_state)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Register_datarep_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Register_datarep_c( datarep, read_conversion_fn,
                          write_conversion_fn, dtype_file_extent_fn, extra_state ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Rsend_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Rsend_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Rsend_c( buf, count, datatype, dest, tag, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Rsend_init_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Rsend_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Rsend_init_c( buf, count, datatype, dest, 
                              tag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Send_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Send_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Send_c( buf, count, datatype, dest, 
                              tag, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Send_init_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Send_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Send_init_c( buf, count, datatype, dest, 
                              tag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Sendrecv_c(const void* sendbuf, MPI_Count sendcount,
    MPI_Datatype sendtype, int dest, int sendtag, void* recvbuf,
    MPI_Count recvcount, MPI_Datatype recvtype, int source, int
    recvtag, MPI_Comm comm, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Sendrecv_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Sendrecv_c( sendbuf, sendcount, sendtype, dest, 
                              sendtag, recvbuf, recvcount, recvtype,
                              source, recvtag, comm, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Sendrecv_replace_c(void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int sendtag, int source, int recvtag,
    MPI_Comm comm, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Sendrecv_replace_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Sendrecv_replace_c( buf, count, datatype, dest,
                              sendtag, source, recvtag, comm, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_call_errhandler(MPI_Session session, int errorcode)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_call_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_call_errhandler( session, errorcode ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_create_errhandler(MPI_Session_errhandler_function*
    session_errhandler_fn, MPI_Errhandler* errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_create_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_create_errhandler( session_errhandler_fn, errhandler ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_finalize(MPI_Session* session)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_finalize()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_finalize( session ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_get_errhandler(MPI_Session session, MPI_Errhandler*
    errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_get_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_get_errhandler( session, errhandler ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_get_info(MPI_Session session, MPI_Info* info_used)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_get_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_get_info( session, info_used ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_get_nth_pset(MPI_Session session, MPI_Info info, int
    n, int* pset_len, char* pset_name)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_get_nth_pset()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_get_nth_pset( session, info, n, pset_len, pset_name ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_get_num_psets(MPI_Session session, MPI_Info info,
    int* npset_names)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_get_num_psets()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_get_num_psets( session, info, npset_names ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_get_pset_info(MPI_Session session, const char*
    pset_name, MPI_Info* info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_get_pset_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_get_pset_info( session, pset_name, info ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_init(MPI_Info info, MPI_Errhandler errhandler,
    MPI_Session* session)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_init()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_init( info, errhandler, session ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Session_set_errhandler(MPI_Session session, MPI_Errhandler
    errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Session_set_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Session_set_errhandler( session, errhandler ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ssend_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ssend_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ssend_c( buf, count, datatype, dest, tag, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Ssend_init_c(const void* buf, MPI_Count count, MPI_Datatype
    datatype, int dest, int tag, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ssend_init_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ssend_init_c( buf, count, datatype, dest, tag, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Status_set_elements_c(MPI_Status* status, MPI_Datatype
    datatype, MPI_Count count)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Status_set_elements_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Status_set_elements_c( status, datatype, count ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_contiguous_c(MPI_Count count, MPI_Datatype oldtype,
    MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_contiguous_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_contiguous_c( count, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_create_darray_c(int size, int rank, int ndims, const
    MPI_Count array_of_gsizes[], const int array_of_distribs[],
    const int array_of_dargs[], const int array_of_psizes[], int
    order, MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_darray_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_darray_c( size, rank, ndims, array_of_gsizes,
                                array_of_distribs, array_of_dargs, array_of_psizes,
                                order, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_create_hindexed_c(MPI_Count count, const MPI_Count
    array_of_blocklengths[], const MPI_Count
    array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype*
    newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hindexed_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_hindexed_c( count, array_of_blocklengths,
                                array_of_displacements, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_create_indexed_block_c(MPI_Count count, MPI_Count
    blocklength, const MPI_Count array_of_displacements[],
    MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_indexed_block_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_indexed_block_c( count, blocklength,
                                array_of_displacements, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_create_hindexed_block_c(MPI_Count count, MPI_Count
    blocklength, const MPI_Count array_of_displacements[],
    MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hindexed_block_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_hindexed_block_c( count, blocklength,
                                array_of_displacements, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_create_hvector_c(MPI_Count count, MPI_Count
    blocklength, MPI_Count stride, MPI_Datatype oldtype,
    MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hvector_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_hvector_c( count, blocklength,
                                stride, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_create_resized_c(MPI_Datatype oldtype, MPI_Count lb,
    MPI_Count extent, MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_resized_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_resized_c( oldtype, lb, extent, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_create_struct_c(MPI_Count count, const MPI_Count
    array_of_blocklengths[], const MPI_Count
    array_of_displacements[], const MPI_Datatype array_of_types[],
    MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_struct_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_struct_c( count, array_of_blocklengths, 
                                  array_of_displacements, array_of_types, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_create_subarray_c(int ndims, const MPI_Count
    array_of_sizes[], const MPI_Count array_of_subsizes[], const
    MPI_Count array_of_starts[], int order, MPI_Datatype oldtype,
    MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_subarray_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_subarray_c( ndims, array_of_sizes, array_of_subsizes,
                                array_of_starts, order, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_get_contents_c(MPI_Datatype datatype, MPI_Count
    max_integers, MPI_Count max_addresses, MPI_Count
    max_large_counts, MPI_Count max_datatypes, int
    array_of_integers[], MPI_Aint array_of_addresses[], MPI_Count
    array_of_large_counts[], MPI_Datatype array_of_datatypes[])
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_contents_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_contents_c( datatype, max_integers, max_addresses,
                                max_large_counts, max_datatypes, array_of_integers,
                                array_of_addresses, array_of_large_counts,
                                array_of_datatypes ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_get_envelope_c(MPI_Datatype datatype, MPI_Count*
    num_integers, MPI_Count* num_addresses, MPI_Count*
    num_large_counts, MPI_Count* num_datatypes, int* combiner)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_envelope_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_envelope_c( datatype, num_integers, num_addresses,
                                num_large_counts, num_datatypes, combiner ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_get_extent_c(MPI_Datatype datatype, MPI_Count* lb,
    MPI_Count* extent)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_extent_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_extent_c( datatype, lb, extent ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_get_true_extent_c(MPI_Datatype datatype, MPI_Count*
    true_lb, MPI_Count* true_extent)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_true_extent_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_true_extent_c( datatype, true_lb, true_extent ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_indexed_c(MPI_Count count, const MPI_Count
    array_of_blocklengths[], const MPI_Count
    array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype*
    newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_indexed_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_indexed_c( count, array_of_blocklengths, array_of_displacements,
                                  oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_size_c(MPI_Datatype datatype, MPI_Count* size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_size_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_size_c( datatype, size ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Type_vector_c(MPI_Count count, MPI_Count blocklength,
    MPI_Count stride, MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_vector_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_vector_c( count, blocklength, stride, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Unpack_c(const void* inbuf, MPI_Count insize, MPI_Count*
    position, void* outbuf, MPI_Count outcount, MPI_Datatype
    datatype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Unpack_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Unpack_c( inbuf, insize, position, outbuf, outcount,
                            datatype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Unpack_external_c(const char datarep[], const void* inbuf,
    MPI_Count insize, MPI_Count* position, void* outbuf, MPI_Count
    outcount, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Unpack_external_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Unpack_external_c( datarep, inbuf, insize, position,
                            outbuf, outcount, datatype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Win_allocate_c(MPI_Aint size, MPI_Aint disp_unit, MPI_Info
    info, MPI_Comm comm, void* baseptr, MPI_Win* win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_allocate_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_allocate_c( size, disp_unit, info, comm, baseptr, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Win_allocate_shared_c(MPI_Aint size, MPI_Aint disp_unit,
    MPI_Info info, MPI_Comm comm, void* baseptr, MPI_Win* win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_allocate_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_allocate_c( size, disp_unit, info, comm, baseptr, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int PI_Win_create_c(void* base, MPI_Aint size, MPI_Aint disp_unit,
    MPI_Info info, MPI_Comm comm, MPI_Win* win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "PI_Win_create_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_create_c( base, size, disp_unit, info, comm, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}

int MPI_Win_shared_query_c(MPI_Win win, int rank, MPI_Aint* size,
    MPI_Aint* disp_unit, void* baseptr)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_shared_query_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_shared_query_c( win, rank, size, disp_unit, baseptr ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue;
}