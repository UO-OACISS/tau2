#include <mpi.h>
#include <TAU.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <Profile/TauUtil.h>
#include <Profile/TauEnv.h>

//https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf

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

//https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Finalized.3.html
//MPI_SESSION_
//MPI_SESSION_{. . . }_ERRHANDLER
//MPI_SESSION_GET_
