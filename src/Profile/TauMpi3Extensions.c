#include <mpi.h>
#include <TAU.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <Profile/TauUtil.h>
#include <Profile/TauEnv.h>

//https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf page 788-

int MPI_Get_library_version ( char *version, int *resultlen )
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_library_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_library_version( version, resultlen ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Type_size_x ( MPI_Datatype datatype, MPI_Count *size )
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_size_x()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_size_x( datatype, size ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Type_get_extent_x ( MPI_Datatype datatype, MPI_Count* lb, MPI_Count* extent )
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_extent_x()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_extent_x( datatype, lb, extent ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Get_elements_x ( const MPI_Status* status, MPI_Datatype datatype, MPI_Count* count )
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_elements_x()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_elements_x( status, datatype, count ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Type_get_true_extent_x ( MPI_Datatype datatype, MPI_Count* true_lb, MPI_Count* true_extent )
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_true_extent_x()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_true_extent_x( datatype, true_lb, true_extent ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Status_set_elements_x ( MPI_Status* status, MPI_Datatype datatype, MPI_Count count )
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Status_set_elements_x()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Status_set_elements_x( status, datatype, count ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Mprobe ( int source, int tag, MPI_Comm comm, MPI_Message*
                                      message, MPI_Status* status )
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Mprobe()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Mprobe( source, tag, comm, message, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Improbe ( int source, int tag, MPI_Comm comm, int* flag,
                          MPI_Message* message, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Improbe()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Improbe( source, tag, comm, flag, message, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Mrecv(void* buf, int count, MPI_Datatype datatype,
                    MPI_Message* message, MPI_Status* status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Mrecv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Mrecv( buf, count, datatype, message, status ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}


int MPI_Imrecv(void* buf, int count, MPI_Datatype datatype,
                      MPI_Message* message, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Imrecv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Imrecv( buf, count, datatype, message, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}


int MPI_Type_create_hindexed_block(int count, int blocklength, const
            MPI_Aint array_of_displacements[], MPI_Datatype oldtype,
            MPI_Datatype* newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hindexed_block()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_hindexed_block( count, blocklength, array_of_displacements, oldtype, newtype ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ibarrier(MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ibarrier()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ibarrier( comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ibcast(void* buffer, int count, MPI_Datatype datatype, int
                root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ibcast()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ibcast( buffer, count, datatype, root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Igather(const void* sendbuf, int sendcount, MPI_Datatype
                sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Igather()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Igather( sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                            root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Igatherv(const void* sendbuf, int sendcount, MPI_Datatype
    sendtype, void* recvbuf, const int recvcounts[], const int
    displs[], MPI_Datatype recvtype, int root, MPI_Comm comm,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Igatherv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Igatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                            recvtype, root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iscatter(const void* sendbuf, int sendcount, MPI_Datatype
    sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iscatter()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iscatter( sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iscatterv(const void* sendbuf, const int sendcounts[], const
    int displs[], MPI_Datatype sendtype, void* recvbuf, int
    recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iscatterv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iscatterv( sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount,
                            recvtype, root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iallgather(const void* sendbuf, int sendcount, MPI_Datatype
    sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iallgather()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iallgather( sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iallgatherv(const void* sendbuf, int sendcount, MPI_Datatype
    sendtype, void* recvbuf, const int recvcounts[], const int
    displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iallgatherv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iallgatherv( sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                            displs, recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ialltoall(const void* sendbuf, int sendcount, MPI_Datatype
    sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ialltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ialltoall( sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ialltoallv(const void* sendbuf, const int sendcounts[],
    const int sdispls[], MPI_Datatype sendtype, void* recvbuf, const
    int recvcounts[], const int rdispls[], MPI_Datatype recvtype,
    MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ialltoallv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ialltoallv( sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
                            rdispls, recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ialltoallw(const void* sendbuf, const int sendcounts[],
    const int sdispls[], const MPI_Datatype sendtypes[], void*
    recvbuf, const int recvcounts[], const int rdispls[], const
    MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ialltoallw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ialltoallw( sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
                            rdispls, recvtypes, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iallreduce(const void* sendbuf, void* recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iallreduce()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iallreduce( sendbuf, recvbuf, count, datatype,
                            op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ireduce(const void* sendbuf, void* recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ireduce()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ireduce( sendbuf, recvbuf, count, datatype,
                            op, root, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

//MPI_Reduce_scatter_block should be from MPI-2

int MPI_Reduce_scatter_block(const void* sendbuf, void* recvbuf, int
    recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_block()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Reduce_scatter_block( sendbuf, recvbuf, recvcount,
                           datatype, op, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ireduce_scatter_block(const void* sendbuf, void* recvbuf,
    int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_scatter_block()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ireduce_scatter_block( sendbuf, recvbuf, recvcount,
                           datatype, op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ireduce_scatter(const void* sendbuf, void* recvbuf, const
    int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm
    comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_scatter()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ireduce_scatter( sendbuf, recvbuf, recvcounts,
                           datatype, op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iscan(const void* sendbuf, void* recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iscan()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iscan( sendbuf, recvbuf, count,
                           datatype, op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Iexscan(const void* sendbuf, void* recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Iexscan()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Iexscan( sendbuf, recvbuf, count,
                           datatype, op, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm*
    newcomm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_dup_with_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_dup_with_info( comm, info, newcomm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Comm_set_info(MPI_Comm comm, MPI_Info info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_set_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_set_info( comm, info ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Comm_get_info(MPI_Comm comm, MPI_Info* info_used)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_get_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_get_info( comm, info_used ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Win_set_info(MPI_Win win, MPI_Info info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_set_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_set_info( win, info ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Win_get_info(MPI_Win win, MPI_Info* info_used)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_get_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_get_info( win, info_used ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Comm_idup(MPI_Comm comm, MPI_Comm* newcomm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_idup()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_idup( comm, newcomm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm* newcomm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_create_group()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_create_group( comm, group, tag, newcomm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm* newcomm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_split_type()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_split_type( comm, split_type, key, info, newcomm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Neighbor_allgather(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgather()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_allgather( sendbuf, sendcount, sendtype, recvbuf, 
                                      recvcount, recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Neighbor_allgatherv(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
    const int displs[], MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgatherv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_allgatherv( sendbuf, sendcount, sendtype, recvbuf, 
                                      recvcounts, displs, recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Neighbor_alltoall(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoall( sendbuf, sendcount, sendtype, recvbuf, 
                                      recvcount, recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Neighbor_alltoallv(const void* sendbuf, const int
    sendcounts[], const int sdispls[], MPI_Datatype sendtype, void*
    recvbuf, const int recvcounts[], const int rdispls[],
    MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoallv( sendbuf, sendcounts, sdispls, sendtype, recvbuf, 
                                      recvcounts, rdispls, recvtype, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Neighbor_alltoallw(const void* sendbuf, const int
    sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype
    sendtypes[], void* recvbuf, const int recvcounts[], const
    MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm
    comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Neighbor_alltoallw( sendbuf, sendcounts, sdispls, sendtypes, recvbuf, 
                                      recvcounts, rdispls, recvtypes, comm ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ineighbor_allgather(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_allgather()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_allgather( sendbuf, sendcount, sendtype, recvbuf, 
                                      recvcount, recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ineighbor_allgatherv(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
    const int displs[], MPI_Datatype recvtype, MPI_Comm comm,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_allgatherv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_allgatherv( sendbuf, sendcount, sendtype, recvbuf, 
                                      recvcounts, displs, recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ineighbor_alltoall(const void* sendbuf, int sendcount,
    MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoall()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_alltoall( sendbuf, sendcount, sendtype, recvbuf, 
                                      recvcount, recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ineighbor_alltoallv(const void* sendbuf, const int
    sendcounts[], const int sdispls[], MPI_Datatype sendtype, void*
    recvbuf, const int recvcounts[], const int rdispls[],
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoallv()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_alltoallv( sendbuf, sendcounts, sdispls, sendtype, recvbuf, 
                                      recvcounts, rdispls, recvtype, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Ineighbor_alltoallw(const void* sendbuf, const int
    sendcounts[], const MPI_Aint sdispls[], const MPI_Datatype
    sendtypes[], void* recvbuf, const int recvcounts[], const
    MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm
    comm, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoallw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Ineighbor_alltoallw( sendbuf, sendcounts, sdispls, sendtypes, recvbuf, 
                                      recvcounts, rdispls, recvtypes, comm, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int
    sources[], int sourceweights[], int maxoutdegree, int
    destinations[], int destweights[])
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Dist_graph_neighbors()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Dist_graph_neighbors( comm, maxindegree, sources, sourceweights,
                                        maxoutdegree, destinations, destweights ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree,
    const int sources[], const int sourceweights[], int outdegree,
    const int destinations[], const int destweights[], MPI_Info
    info, int reorder, MPI_Comm* comm_dist_graph)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Dist_graph_create_adjacent()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Dist_graph_create_adjacent( comm_old, indegree, sources, sourceweights,
                                        outdegree, destinations, destweights, info,
                                        reorder, comm_dist_graph ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Rput(const void* origin_addr, int origin_count, MPI_Datatype
    origin_datatype, int target_rank, MPI_Aint target_disp, int
    target_count, MPI_Datatype target_datatype, MPI_Win win,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Rput()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Rput( origin_addr, origin_count, origin_datatype,
                        target_rank, target_disp, target_count,
                        target_datatype, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Rget(void* origin_addr, int origin_count, MPI_Datatype
    origin_datatype, int target_rank, MPI_Aint target_disp, int
    target_count, MPI_Datatype target_datatype, MPI_Win win,
    MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Rget()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Rget( origin_addr, origin_count, origin_datatype,
                        target_rank, target_disp, target_count,
                        target_datatype, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Raccumulate(const void* origin_addr, int origin_count,
    MPI_Datatype origin_datatype, int target_rank, MPI_Aint
    target_disp, int target_count, MPI_Datatype target_datatype,
    MPI_Op op, MPI_Win win, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Raccumulate()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Raccumulate( origin_addr, origin_count, origin_datatype,
                        target_rank, target_disp, target_count,
                        target_datatype, op, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Get_accumulate(const void* origin_addr, int origin_count,
    MPI_Datatype origin_datatype, void* result_addr, int
    result_count, MPI_Datatype result_datatype, int target_rank,
    MPI_Aint target_disp, int target_count, MPI_Datatype
    target_datatype, MPI_Op op, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_accumulate()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_accumulate( origin_addr, origin_count, origin_datatype,
                        result_addr, result_count, result_datatype,
                        target_rank, target_disp, target_count,
                        target_datatype, op, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Rget_accumulate(const void* origin_addr, int origin_count,
    MPI_Datatype origin_datatype, void* result_addr, int
    result_count, MPI_Datatype result_datatype, int target_rank,
    MPI_Aint target_disp, int target_count, MPI_Datatype
    target_datatype, MPI_Op op, MPI_Win win, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Rget_accumulate()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Rget_accumulate( origin_addr, origin_count, origin_datatype,
                        result_addr, result_count, result_datatype,
                        target_rank, target_disp, target_count,
                        target_datatype, op, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Fetch_and_op(const void* origin_addr, void* result_addr,
    MPI_Datatype datatype, int target_rank, MPI_Aint target_disp,
    MPI_Op op, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Fetch_and_op()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Fetch_and_op( origin_addr, result_addr, datatype, 
                                target_rank, target_disp, op, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Compare_and_swap(const void* origin_addr, const void*
    compare_addr, void* result_addr, MPI_Datatype datatype, int
    target_rank, MPI_Aint target_disp, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Compare_and_swap()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Compare_and_swap( origin_addr, compare_addr, result_addr,
                                datatype, target_rank, target_disp, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info,
    MPI_Comm comm, void* baseptr, MPI_Win* win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_allocate()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_allocate( size, disp_unit, info, comm, baseptr, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info
    info, MPI_Comm comm, void* baseptr, MPI_Win* win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_allocate_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_allocate_shared( size, disp_unit, info, comm, baseptr, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win* win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_create_dynamic()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_create_dynamic( info, comm, win ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}
