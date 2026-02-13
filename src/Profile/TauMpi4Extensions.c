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
  retvalue = PMPI_Info_get_string(  info, key, buflen, value, flag ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_Info_create_env(int argc, char **argv, MPI_Info* info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_create_env()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_create_env(  argc, argv, info ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iread_at_all(MPI_File fh, MPI_Offset offset, void* buf,
    int count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_at_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_at_all(  fh, offset, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iwrite_at_all(MPI_File fh, MPI_Offset offset, const
    void* buf, int count, MPI_Datatype datatype, MPI_Request*
    request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_at_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_at_all(  fh, offset, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iread_all(MPI_File fh, void* buf, int count,
    MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_all(  fh, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

int MPI_File_iwrite_all(MPI_File fh, const void* buf, int count,
    MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_all(  fh, buf, count, datatype, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

MPI_Aint MPI_Aint_add(MPI_Aint base, MPI_Aint disp)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Aint_add()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Aint_add(  base, disp ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

MPI_Aint MPI_Aint_diff(MPI_Aint addr1, MPI_Aint addr2)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Aint_diff()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Aint_diff(  addr1, addr2 ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

//Large count functions

int MPI_Accumulate_c(const void* origin_addr, MPI_Count
    origin_count, MPI_Datatype origin_datatype, int target_rank,
    MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype
    target_datatype, MPI_Op op, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Accumulate_c()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Accumulate_c(  origin_addr, origin_count, origin_datatype,
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
  retvalue = PMPI_Raccumulate_c(  origin_addr, origin_count, origin_datatype,
                                target_rank, target_disp, target_count, 
                                target_datatype, op, win, request ) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}


