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