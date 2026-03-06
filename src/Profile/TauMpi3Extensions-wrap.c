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

/******************************************************
***      MPI_Get_library_version wrapper function 
******************************************************/
int MPI_Get_library_version(char *version, int * resultlen)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Get_library_version()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Get_library_version(version, resultlen);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Get_library_version wrapper function (uppercase Fortran)
******************************************************/
void MPI_GET_LIBRARY_VERSION(char * version, MPI_Fint * resultlen, MPI_Fint * ierr)
{

  *ierr = MPI_Get_library_version( version, resultlen);
  return ;
}

/******************************************************
***      MPI_Get_library_version wrapper function (lowercase)
******************************************************/
void mpi_get_library_version(char * version, MPI_Fint * resultlen, MPI_Fint * ierr)
{
  MPI_GET_LIBRARY_VERSION(version, resultlen, ierr);
  return ;
}

/******************************************************
***      MPI_Get_library_version wrapper function (lowercase_)
******************************************************/
void mpi_get_library_version_(char * version, MPI_Fint * resultlen, MPI_Fint * ierr)
{
  MPI_GET_LIBRARY_VERSION(version, resultlen, ierr);
  return ;
}

/******************************************************
***      MPI_Get_library_version wrapper function (lowercase__)
******************************************************/
void mpi_get_library_version__(char * version, MPI_Fint * resultlen, MPI_Fint * ierr)
{
  MPI_GET_LIBRARY_VERSION(version, resultlen, ierr);
  return ;
}


/******************************************************
***      MPI_Type_size_x wrapper function 
******************************************************/
int MPI_Type_size_x(MPI_Datatype datatype, MPI_Count * size)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_size_x()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Type_size_x(datatype, size);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_size_x wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_SIZE_X(MPI_Fint * datatype, MPI_Count * size, MPI_Fint * ierr)
{

  *ierr = MPI_Type_size_x( MPI_Type_f2c(*datatype), size);
  return ;
}

/******************************************************
***      MPI_Type_size_x wrapper function (lowercase)
******************************************************/
void mpi_type_size_x(MPI_Fint * datatype, MPI_Count * size, MPI_Fint * ierr)
{
  MPI_TYPE_SIZE_X(datatype, size, ierr);
  return ;
}

/******************************************************
***      MPI_Type_size_x wrapper function (lowercase_)
******************************************************/
void mpi_type_size_x_(MPI_Fint * datatype, MPI_Count * size, MPI_Fint * ierr)
{
  MPI_TYPE_SIZE_X(datatype, size, ierr);
  return ;
}

/******************************************************
***      MPI_Type_size_x wrapper function (lowercase__)
******************************************************/
void mpi_type_size_x__(MPI_Fint * datatype, MPI_Count * size, MPI_Fint * ierr)
{
  MPI_TYPE_SIZE_X(datatype, size, ierr);
  return ;
}


/******************************************************
***      MPI_Type_get_extent_x wrapper function 
******************************************************/
int MPI_Type_get_extent_x(MPI_Datatype datatype, MPI_Count * lb, MPI_Count * extent)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_get_extent_x()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Type_get_extent_x(datatype, lb, extent);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_get_extent_x wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_GET_EXTENT_X(MPI_Fint * datatype, MPI_Count * lb, MPI_Count * extent, MPI_Fint * ierr)
{

  *ierr = MPI_Type_get_extent_x(MPI_Type_f2c(*datatype), lb, extent);
  return ;
}

/******************************************************
***      MPI_Type_get_extent_x wrapper function (lowercase)
******************************************************/
void mpi_type_get_extent_x(MPI_Fint * datatype, MPI_Count * lb, MPI_Count * extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT_X(datatype, lb, extent, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_extent_x wrapper function (lowercase_)
******************************************************/
void mpi_type_get_extent_x_(MPI_Fint * datatype, MPI_Count * lb, MPI_Count * extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT_X(datatype, lb, extent, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_extent_x wrapper function (lowercase__)
******************************************************/
void mpi_type_get_extent_x__(MPI_Fint * datatype, MPI_Count * lb, MPI_Count * extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT_X(datatype, lb, extent, ierr);
  return ;
}


/******************************************************
***      MPI_Get_elements_x wrapper function 
******************************************************/
int MPI_Get_elements_x(TAU_MPICH3_CONST MPI_Status * status, MPI_Datatype datatype, MPI_Count* count)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Get_elements_x()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Get_elements_x(status, datatype, count);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Get_elements_x wrapper function (uppercase Fortran)
******************************************************/
void MPI_GET_ELEMENTS_X(MPI_Fint * status, MPI_Fint * datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_Status local_status; 
  MPI_Status_f2c(status, &local_status);
  *ierr = MPI_Get_elements_x( &local_status, MPI_Type_f2c(*datatype), count);
  MPI_Status_c2f(&local_status, status);
  return ;
}

/******************************************************
***      MPI_Get_elements_x wrapper function (lowercase)
******************************************************/
void mpi_get_elements_x(MPI_Fint * status, MPI_Fint * datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_ELEMENTS_X(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Get_elements_x wrapper function (lowercase_)
******************************************************/
void mpi_get_elements_x_(MPI_Fint * status, MPI_Fint * datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_ELEMENTS_X(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Get_elements_x wrapper function (lowercase__)
******************************************************/
void mpi_get_elements_x__(MPI_Fint * status, MPI_Fint * datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_ELEMENTS_X(status, datatype, count, ierr);
  return ;
}


/******************************************************
***      MPI_Type_get_true_extent_x wrapper function 
******************************************************/
int MPI_Type_get_true_extent_x(MPI_Datatype datatype, MPI_Count* true_lb, MPI_Count* true_extent)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_get_true_extent_x()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Type_get_true_extent_x(datatype, true_lb, true_extent);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_get_true_extent_x wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_GET_TRUE_EXTENT_X(MPI_Fint * datatype, MPI_Count* true_lb, MPI_Count* true_extent, MPI_Fint * ierr)
{

  *ierr = MPI_Type_get_true_extent_x(MPI_Type_f2c(*datatype), true_lb, true_extent);
  return ;
}

/******************************************************
***      MPI_Type_get_true_extent_x wrapper function (lowercase)
******************************************************/
void mpi_type_get_true_extent_x(MPI_Fint * datatype, MPI_Count* true_lb, MPI_Count* true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT_X(datatype, true_lb, true_extent, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_true_extent_x wrapper function (lowercase_)
******************************************************/
void mpi_type_get_true_extent_x_(MPI_Fint * datatype, MPI_Count* true_lb, MPI_Count* true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT_X(datatype, true_lb, true_extent, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_true_extent_x wrapper function (lowercase__)
******************************************************/
void mpi_type_get_true_extent_x__(MPI_Fint * datatype, MPI_Count* true_lb, MPI_Count* true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT_X(datatype, true_lb, true_extent, ierr);
  return ;
}


/******************************************************
***      MPI_Status_set_elements_x wrapper function 
******************************************************/
int MPI_Status_set_elements_x(MPI_Status* status, MPI_Datatype datatype, MPI_Count count)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Status_set_elements_x()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Status_set_elements_x(status, datatype, count);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Status_set_elements_x wrapper function (uppercase Fortran)
******************************************************/
void MPI_STATUS_SET_ELEMENTS_X(MPI_Fint * status, MPI_Fint * datatype, MPI_Count * count, MPI_Fint * ierr)
{
  MPI_Status local_status; 
  MPI_Status_f2c(status, &local_status);
  *ierr = MPI_Status_set_elements_x(&local_status, MPI_Type_f2c(*datatype),
                                    *count);
  MPI_Status_c2f(&local_status, status);
  return ;
}

/******************************************************
***      MPI_Status_set_elements_x wrapper function (lowercase)
******************************************************/
void mpi_status_set_elements_x(MPI_Fint * status, MPI_Fint * datatype, MPI_Count * count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS_X(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Status_set_elements_x wrapper function (lowercase_)
******************************************************/
void mpi_status_set_elements_x_(MPI_Fint * status, MPI_Fint * datatype, MPI_Count * count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS_X(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Status_set_elements_x wrapper function (lowercase__)
******************************************************/
void mpi_status_set_elements_x__(MPI_Fint * status, MPI_Fint * datatype, MPI_Count * count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS_X(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Mprobe wrapper function 
******************************************************/
int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message* message, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Mprobe()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Mprobe(source, tag, comm, message, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}


/******************************************************
***      MPI_Mprobe wrapper function (uppercase Fortran)
******************************************************/
void MPI_MPROBE(MPI_Fint * source, MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * message, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_Status local_status; 
  MPI_Message local_message;
  *ierr = MPI_Mprobe(*source, *tag,
                      MPI_Comm_f2c(*comm), &message, &local_status);
  MPI_Status_c2f(&local_status, status);
  *message = MPI_Message_c2f(local_message);
  return ;
}

/******************************************************
***      MPI_Mprobe wrapper function (lowercase)
******************************************************/
void mpi_mprobe(MPI_Fint * source, MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * message, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_MPROBE(source, tag, comm, message, status, ierr);
  return ;
}

/******************************************************
***      MPI_Mprobe wrapper function (lowercase_)
******************************************************/
void mpi_mprobe_(MPI_Fint * source, MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * message, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_MPROBE(source, tag, comm, message, status, ierr);
  return ;
}

/******************************************************
***      MPI_Mprobe wrapper function (lowercase__)
******************************************************/
void mpi_mprobe__(MPI_Fint * source, MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * message, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_MPROBE(source, tag, comm, message, status, ierr);
  return ;
}


/******************************************************
***      MPI_Improbe wrapper function 
******************************************************/
int MPI_Improbe(int source, int tag, MPI_Comm comm, int* flag, MPI_Message* message, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Improbe()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Improbe(source, tag, comm, flag, message, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Improbe wrapper function (uppercase Fortran)
******************************************************/
void MPI_IMPROBE(MPI_Fint * source, MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * flag, MPI_Fint * message, MPI_Fint * status,
MPI_Fint * ierr)
{
  MPI_Status local_status; 
  MPI_Message local_message;
  *ierr = MPI_Improbe( *source, *tag, 
                       MPI_Comm_f2c(*comm), flag, &message, &local_status);
  MPI_Status_c2f(&local_status, status);
  *message = MPI_Message_c2f(local_message);
  return ;
}

/******************************************************
***      MPI_Improbe wrapper function (lowercase)
******************************************************/
void mpi_improbe(MPI_Fint * source, MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * flag, MPI_Fint * message, MPI_Fint * status,
MPI_Fint * ierr)
{
  MPI_IMPROBE(source, tag, comm, flag, message, status, ierr);
  return ;
}

/******************************************************
***      MPI_Improbe wrapper function (lowercase_)
******************************************************/
void mpi_improbe_(MPI_Fint * source, MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * flag, MPI_Fint * message, MPI_Fint * status,
MPI_Fint * ierr)
{
  MPI_IMPROBE(source, tag, comm, flag, message, status, ierr);
  return ;
}

/******************************************************
***      MPI_Improbe wrapper function (lowercase__)
******************************************************/
void mpi_improbe__(MPI_Fint * source, MPI_Fint * tag, MPI_Fint * comm, MPI_Fint * flag, MPI_Fint * message, MPI_Fint * status,
MPI_Fint * ierr)
{
  MPI_IMPROBE(source, tag, comm, flag, message, status, ierr);
  return ;
}


/******************************************************
***      MPI_Mrecv wrapper function 
******************************************************/
int MPI_Mrecv(void* buf, int count, MPI_Datatype datatype, MPI_Message* message, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Mrecv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Mrecv(buf, count, datatype, message, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Mrecv wrapper function (uppercase Fortran)
******************************************************/
void MPI_MRECV(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * message, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_Status local_status; 
  MPI_Message local_message;
  local_message = MPI_Message_f2c(*message);
  *ierr = MPI_Mrecv(buf, *count, MPI_Type_f2c(*datatype),
                    &local_message, &local_status);
  MPI_Status_c2f(&local_status, status);
  *message = MPI_Message_c2f(local_message);
  return ;
}

/******************************************************
***      MPI_Mrecv wrapper function (lowercase)
******************************************************/
void mpi_mrecv(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * message, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_MRECV(buf, count, datatype, message, status, ierr);
  return ;
}

/******************************************************
***      MPI_Mrecv wrapper function (lowercase_)
******************************************************/
void mpi_mrecv_(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * message, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_MRECV(buf, count, datatype, message, status, ierr);
  return ;
}

/******************************************************
***      MPI_Mrecv wrapper function (lowercase__)
******************************************************/
void mpi_mrecv__(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * message, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_MRECV(buf, count, datatype, message, status, ierr);
  return ;
}


/******************************************************
***      MPI_Imrecv wrapper function 
******************************************************/
int MPI_Imrecv(void* buf, int count, MPI_Datatype datatype, MPI_Message* message, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Imrecv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Imrecv(buf, count, datatype, message, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Imrecv wrapper function (uppercase Fortran)
******************************************************/
void MPI_IMRECV(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * message, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_Message local_message;
  MPI_Request local_request;
  local_message = MPI_Message_f2c(*message);
  *ierr = MPI_Imrecv(buf, *count, MPI_Type_f2c(*datatype),
    &local_message, &local_request);
  *message = MPI_Message_c2f(local_message);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Imrecv wrapper function (lowercase)
******************************************************/
void mpi_imrecv(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * message, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IMRECV(buf, count, datatype, message, request, ierr);
  return ;
}

/******************************************************
***      MPI_Imrecv wrapper function (lowercase_)
******************************************************/
void mpi_imrecv_(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * message, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IMRECV(buf, count, datatype, message, request, ierr);
  return ;
}

/******************************************************
***      MPI_Imrecv wrapper function (lowercase__)
******************************************************/
void mpi_imrecv__(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * message, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IMRECV(buf, count, datatype, message, request, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_block wrapper function 
******************************************************/
int MPI_Type_create_hindexed_block(int count, int blocklength, TAU_MPICH3_CONST MPI_Aint * array_of_displacements, MPI_Datatype oldtype,
MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hindexed_block()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Type_create_hindexed_block(count, blocklength, array_of_displacements, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_hindexed_block wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_HINDEXED_BLOCK(MPI_Fint * count, MPI_Fint * blocklength, MPI_Aint * array_of_displacements, MPI_Fint * oldtype,
MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_create_hindexed_block( *count, *blocklength, array_of_displacements, 
              MPI_Type_f2c(*oldtype), &local_type);
  *newtype = MPI_Type_c2f(local_type);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_block wrapper function (lowercase)
******************************************************/
void mpi_type_create_hindexed_block(MPI_Fint * count, MPI_Fint * blocklength, MPI_Aint * array_of_displacements, MPI_Fint * oldtype,
MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_BLOCK(count, blocklength, array_of_displacements, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_block wrapper function (lowercase_)
******************************************************/
void mpi_type_create_hindexed_block_(MPI_Fint * count, MPI_Fint * blocklength, MPI_Aint * array_of_displacements, MPI_Fint * oldtype,
MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_BLOCK(count, blocklength, array_of_displacements, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_block wrapper function (lowercase__)
******************************************************/
void mpi_type_create_hindexed_block__(MPI_Fint * count, MPI_Fint * blocklength, MPI_Aint * array_of_displacements, MPI_Fint * oldtype,
MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_BLOCK(count, blocklength, array_of_displacements, oldtype, newtype, ierr);
  return ;
}



/******************************************************
***      MPI_Ibarrier wrapper function 
******************************************************/
int MPI_Ibarrier(MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ibarrier()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ibarrier(comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ibarrier wrapper function (uppercase Fortran)
******************************************************/
void MPI_IBARRIER(MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Ibarrier(MPI_Comm_f2c(*comm), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Ibarrier wrapper function (lowercase)
******************************************************/
void mpi_ibarrier(MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IBARRIER(comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ibarrier wrapper function (lowercase_)
******************************************************/
void mpi_ibarrier_(MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IBARRIER(comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ibarrier wrapper function (lowercase__)
******************************************************/
void mpi_ibarrier__(MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IBARRIER(comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Ibcast wrapper function 
******************************************************/
int MPI_Ibcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ibcast()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ibcast(buffer, count, datatype, root, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}
MPI_BOTTOM
/******************************************************
***      MPI_Ibcast wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IBCAST(MPI_Aint * buffer, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request,
MPI_Fint * ierr);

/******************************************************
***      MPI_Ibcast wrapper function (lowercase)
******************************************************/
void mpi_ibcast(MPI_Aint * buffer, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request,
MPI_Fint * ierr);
{
  MPI_IBCAST(buffer, count, datatype, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ibcast wrapper function (lowercase_)
******************************************************/
void mpi_ibcast_(MPI_Aint * buffer, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request,
MPI_Fint * ierr);
{
  MPI_IBCAST(buffer, count, datatype, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ibcast wrapper function (lowercase__)
******************************************************/
void mpi_ibcast__(MPI_Aint * buffer, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request,
MPI_Fint * ierr)
{
  MPI_IBCAST(buffer, count, datatype, root, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Igather wrapper function 
******************************************************/
int MPI_Igather(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Igather()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Igather wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IGATHER(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Igather wrapper function (lowercase)
******************************************************/
void mpi_igather(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)

{
  MPI_IGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Igather wrapper function (lowercase_)
******************************************************/
void mpi_igather_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)

{
  MPI_IGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Igather wrapper function (lowercase__)
******************************************************/
void mpi_igather__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Igatherv wrapper function 
******************************************************/
int MPI_Igatherv(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, TAU_MPICH3_CONST int * recvcounts,
TAU_MPICH3_CONST int * displs, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Igatherv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Igatherv wrapper function (uppercase Fortran)
******************************************************/
void MPI_IGATHERV(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);


/******************************************************
***      MPI_Igatherv wrapper function (lowercase)
******************************************************/
void mpi_igatherv(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Igatherv wrapper function (lowercase_)
******************************************************/
void mpi_igatherv_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Igatherv wrapper function (lowercase__)
******************************************************/
void mpi_igatherv__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    request, ierr);
  return ;
}

!!!
/******************************************************
***      MPI_Iscatter wrapper function 
******************************************************/
int MPI_Iscatter(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iscatter()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iscatter wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_ISCATTER(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);
/******************************************************
***      MPI_Iscatter wrapper function (lowercase)
******************************************************/
void mpi_iscatter(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCATTER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscatter wrapper function (lowercase_)
******************************************************/
void mpi_iscatter_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCATTER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscatter wrapper function (lowercase__)
******************************************************/
void mpi_iscatter__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCATTER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Iscatterv wrapper function 
******************************************************/
int MPI_Iscatterv(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int * sendcounts, TAU_MPICH3_CONST int * displs, MPI_Datatype sendtype,
void* recvbuf, int * recvcount, MPI_Datatype recvtype, int * root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iscatterv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iscatterv wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_ISCATTERV(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Fint * displs, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Fint * recvcount, MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Iscatterv wrapper function (lowercase)
******************************************************/
void mpi_iscatterv(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Fint * displs, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Fint * recvcount, MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCATTERV(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscatterv wrapper function (lowercase_)
******************************************************/
void mpi_iscatterv_(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Fint * displs, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Fint * recvcount, MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCATTERV(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscatterv wrapper function (lowercase__)
******************************************************/
void mpi_iscatterv__(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Fint * displs, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Fint * recvcount, MPI_Fint * recvtype, MPI_Fint * root, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCATTERV(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Iallgather wrapper function 
******************************************************/
int MPI_Iallgather(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iallgather()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iallgather wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IALLGATHER(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Iallgather wrapper function (lowercase)
******************************************************/
void mpi_iallgather(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallgather wrapper function (lowercase_)
******************************************************/
void mpi_iallgather_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallgather wrapper function (lowercase__)
******************************************************/
void mpi_iallgather__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Iallgatherv wrapper function 
******************************************************/
int MPI_Iallgatherv(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, TAU_MPICH3_CONST int * recvcounts,
TAU_MPICH3_CONST int * displs, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iallgatherv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iallgatherv wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IALLGATHERV(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
int * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Iallgatherv wrapper function (lowercase)
******************************************************/
void mpi_iallgatherv(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
int * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallgatherv wrapper function (lowercase_)
******************************************************/
void mpi_iallgatherv_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
int * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallgatherv wrapper function (lowercase__)
******************************************************/
void mpi_iallgatherv__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
int * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Ialltoall wrapper function 
******************************************************/
int MPI_Ialltoall(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ialltoall()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ialltoall wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IALLTOALL(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint - recvbuf, MPI_Fint - recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ialltoall wrapper function (lowercase)
******************************************************/
void mpi_ialltoall(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint - recvbuf, MPI_Fint - recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoall wrapper function (lowercase_)
******************************************************/
void mpi_ialltoall_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint - recvbuf, MPI_Fint - recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoall wrapper function (lowercase__)
******************************************************/
void mpi_ialltoall__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint - recvbuf, MPI_Fint - recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Ialltoallv wrapper function 
******************************************************/
int MPI_Ialltoallv(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int * sendcounts, TAU_MPICH3_CONST int * sdispls, MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST int * recvcounts, TAU_MPICH3_CONST int * rdispls, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ialltoallv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ialltoallv wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IALLTOALLV(MPI_Aint sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint recvtype, MPI_Fint comm, MPI_Fint request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ialltoallv wrapper function (lowercase)
******************************************************/
void mpi_ialltoallv(MPI_Aint sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint recvtype, MPI_Fint comm, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_IALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoallv wrapper function (lowercase_)
******************************************************/
void mpi_ialltoallv_(MPI_Aint sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint recvtype, MPI_Fint comm, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_IALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoallv wrapper function (lowercase__)
******************************************************/
void mpi_ialltoallv__(MPI_Aint sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint recvtype, MPI_Fint comm, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_IALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Ialltoallw wrapper function 
******************************************************/
int MPI_Ialltoallw(TAU_MPICH3_CONST void * sendbuf, TAU_MPICH3_CONST int * sendcounts, TAU_MPICH3_CONST int * sdispls, TAU_MPICH3_CONST MPI_Datatype * sendtypes,
void * recvbuf, TAU_MPICH3_CONST int * recvcounts, TAU_MPICH3_CONST int * rdispls, TAU_MPICH3_CONST MPI_Datatype * recvtypes,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ialltoallw()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ialltoallw wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IALLTOALLW(MPI_Aint sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint * recvtypes, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ialltoallw wrapper function (lowercase)
******************************************************/
void mpi_ialltoallw(MPI_Aint sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint * recvtypes, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoallw wrapper function (lowercase_)
******************************************************/
void mpi_ialltoallw_(MPI_Aint sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint * recvtypes, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoallw wrapper function (lowercase__)
******************************************************/
void mpi_ialltoallw__(MPI_Aint sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint * recvtypes, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Iallreduce wrapper function 
******************************************************/
int MPI_Iallreduce(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iallreduce()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iallreduce wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IALLREDUCE(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Iallreduce wrapper function (lowercase)
******************************************************/
void mpi_iallreduce(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLREDUCE(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallreduce wrapper function (lowercase_)
******************************************************/
void mpi_iallreduce_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLREDUCE(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallreduce wrapper function (lowercase__)
******************************************************/
void mpi_iallreduce__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IALLREDUCE(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Ireduce wrapper function 
******************************************************/
int MPI_Ireduce(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ireduce()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ireduce wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IREDUCE(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * root,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ireduce wrapper function (lowercase)
******************************************************/
void mpi_ireduce(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * root,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce wrapper function (lowercase_)
******************************************************/
void mpi_ireduce_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * root,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce wrapper function (lowercase__)
******************************************************/
void mpi_ireduce__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * root,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);
  return ;
}


//MPI_Reduce_scatter_block should be from MPI-2
/******************************************************
***      MPI_Reduce_scatter_block wrapper function 
******************************************************/
int MPI_Reduce_scatter_block(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_block()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_scatter_block wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_REDUCE_SCATTER_BLOCK(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * ierr);

/******************************************************
***      MPI_Reduce_scatter_block wrapper function (lowercase)
******************************************************/
void mpi_reduce_scatter_block(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block wrapper function (lowercase_)
******************************************************/
void mpi_reduce_scatter_block_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block wrapper function (lowercase__)
******************************************************/
void mpi_reduce_scatter_block__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ireduce_scatter_block wrapper function 
******************************************************/
int MPI_Ireduce_scatter_block(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_scatter_block()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ireduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ireduce_scatter_block wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IREDUCE_SCATTER_BLOCK(MPI_Aint sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ireduce_scatter_block wrapper function (lowercase)
******************************************************/
void mpi_ireduce_scatter_block(MPI_Aint sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_BLOCK(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter_block wrapper function (lowercase_)
******************************************************/
void mpi_ireduce_scatter_block_(MPI_Aint sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_BLOCK(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter_block wrapper function (lowercase__)
******************************************************/
void mpi_ireduce_scatter_block__(MMPI_Aint sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_BLOCK(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Ireduce_scatter wrapper function 
******************************************************/
int MPI_Ireduce_scatter(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, TAU_MPICH3_CONST int * recvcounts, MPI_Datatype datatype, MPI_Op op,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_scatter()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ireduce_scatter wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IREDUCE_SCATTER(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ireduce_scatter wrapper function (lowercase)
******************************************************/
void mpi_ireduce_scatter(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter wrapper function (lowercase_)
******************************************************/
void mpi_ireduce_scatter_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter wrapper function (lowercase__)
******************************************************/
void mpi_ireduce_scatter__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Fint * datatype, MPI_Fint * op,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Iscan wrapper function 
******************************************************/
int MPI_Iscan(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iscan()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iscan wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_ISCAN(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Iscan wrapper function (lowercase)
******************************************************/
void mpi_iscan(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCAN(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscan wrapper function (lowercase_)
******************************************************/
void mpi_iscan_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCAN(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscan wrapper function (lowercase__)
******************************************************/
void mpi_iscan__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_ISCAN(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Iexscan wrapper function 
******************************************************/
int MPI_Iexscan(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iexscan()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Iexscan(sendbuf, recvbuf, count, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iexscan wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_IEXSCAN(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Iexscan wrapper function (lowercase)
******************************************************/
void mpi_iexscan(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IEXSCAN(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iexscan wrapper function (lowercase_)
******************************************************/
void mpi_iexscan_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IEXSCAN(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iexscan wrapper function (lowercase__)
******************************************************/
void mpi_iexscan__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_IEXSCAN(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Comm_dup_with_info wrapper function 
******************************************************/
int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm* newcomm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Comm_dup_with_info()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Comm_dup_with_info(comm, info, newcomm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Comm_dup_with_info wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMM_DUP_WITH_INFO(MPI_Fint * comm, MPI_Fint * info, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_Comm local_newcomm;
  *ierr = MPI_Comm_dup_with_info(MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_newcomm);
  *newcomm = MPI_Comm_c2f(local_newcomm);
  return ;
}

/******************************************************
***      MPI_Comm_dup_with_info wrapper function (lowercase)
******************************************************/
void mpi_comm_dup_with_info(MPI_Fint * comm, MPI_Fint * info, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_DUP_WITH_INFO(comm, info, newcomm, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_dup_with_info wrapper function (lowercase_)
******************************************************/
void mpi_comm_dup_with_info_(MPI_Fint * comm, MPI_Fint * info, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_DUP_WITH_INFO(comm, info, newcomm, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_dup_with_info wrapper function (lowercase__)
******************************************************/
void mpi_comm_dup_with_info__(MPI_Fint * comm, MPI_Fint * info, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_DUP_WITH_INFO(comm, info, newcomm, ierr);
  return ;
}


/******************************************************
***      MPI_Comm_set_info wrapper function 
******************************************************/
int MPI_Comm_set_info(MPI_Comm comm, MPI_Info info)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Comm_set_info()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Comm_set_info(comm, info);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Comm_set_info wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMM_SET_INFO(MPI_Fint * comm, MPI_Fint * info, MPI_Fint * ierr)
{

  *ierr = MPI_Comm_set_info(MPI_Comm_f2c(*comm), MPI_Info_f2c(*info));
  return ;
}

/******************************************************
***      MPI_Comm_set_info wrapper function (lowercase)
******************************************************/
void mpi_comm_set_info(MPI_Fint * comm, MPI_Fint * info, MPI_Fint * ierr)
{
  MPI_COMM_SET_INFO(comm, info, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_set_info wrapper function (lowercase_)
******************************************************/
void mpi_comm_set_info_(MPI_Fint * comm, MPI_Fint * info, MPI_Fint * ierr)
{
  MPI_COMM_SET_INFO(comm, info, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_set_info wrapper function (lowercase__)
******************************************************/
void mpi_comm_set_info__(MPI_Fint * comm, MPI_Fint * info, MPI_Fint * ierr)
{
  MPI_COMM_SET_INFO(comm, info, ierr);
  return ;
}


/******************************************************
***      MPI_Comm_get_info wrapper function 
******************************************************/
int MPI_Comm_get_info(MPI_Comm comm, MPI_Info* info_used)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Comm_get_info()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Comm_get_info(comm, info_used);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Comm_get_info wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMM_GET_INFO(MPI_Fint * comm, MPI_Fint * info_used, MPI_Fint * ierr)
{
  MPI_Info local_info_used;
  *ierr = MPI_Comm_get_info(MPI_Comm_f2c(*comm), local_info_used);
  *info_used = MPI_Info_f2c(local_info_used);
  return ;
}

/******************************************************
***      MPI_Comm_get_info wrapper function (lowercase)
******************************************************/
void mpi_comm_get_info(MPI_Fint * comm, MPI_Fint * info_used, MPI_Fint * ierr)
{
  MPI_COMM_GET_INFO(comm, info_used, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_get_info wrapper function (lowercase_)
******************************************************/
void mpi_comm_get_info_(MPI_Fint * comm, MPI_Fint * info_used, MPI_Fint * ierr)
{
  MPI_COMM_GET_INFO(comm, info_used, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_get_info wrapper function (lowercase__)
******************************************************/
void mpi_comm_get_info__(MPI_Fint * comm, MPI_Fint * info_used, MPI_Fint * ierr)
{
  MPI_COMM_GET_INFO(comm, info_used, ierr);
  return ;
}


/******************************************************
***      MPI_Win_set_info wrapper function 
******************************************************/
int MPI_Win_set_info(MPI_Win win, MPI_Info info)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_set_info()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_set_info(win, info);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_set_info wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_SET_INFO(MPI_Fint * win, MPI_Fint * info, MPI_Fint * ierr)
{

  *ierr = MPI_Win_set_info(MPI_Win_f2c(*win), MPI_Info_f2c(*info));
  return ;
}

/******************************************************
***      MPI_Win_set_info wrapper function (lowercase)
******************************************************/
void mpi_win_set_info(MPI_Fint * win, MPI_Fint * info, MPI_Fint * ierr)
{
  MPI_WIN_SET_INFO(win, info, ierr);
  return ;
}

/******************************************************
***      MPI_Win_set_info wrapper function (lowercase_)
******************************************************/
void mpi_win_set_info_(MPI_Fint * win, MPI_Fint * info, MPI_Fint * ierr)
{
  MPI_WIN_SET_INFO(win, info, ierr);
  return ;
}

/******************************************************
***      MPI_Win_set_info wrapper function (lowercase__)
******************************************************/
void mpi_win_set_info__(MPI_Fint * win, MPI_Fint * info, MPI_Fint * ierr)
{
  MPI_WIN_SET_INFO(win, info, ierr);
  return ;
}


/******************************************************
***      MPI_Win_get_info wrapper function 
******************************************************/
int MPI_Win_get_info(MPI_Win win, MPI_Info* info_used)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_get_info()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_get_info(win, info_used);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_get_info wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_GET_INFO(MPI_Fint * win, MPI_Fint * info_used, MPI_Fint * ierr)
{
  MPI_Info local_info_used;
  *ierr = MPI_Win_get_info(MPI_Win_f2c(*win), local_info_used);
  *info_used = MPI_Info_f2c(local_info_used);
  return ;
}

/******************************************************
***      MPI_Win_get_info wrapper function (lowercase)
******************************************************/
void mpi_win_get_info(MPI_Fint * win, MPI_Fint * info_used, MPI_Fint * ierr)
{
  MPI_WIN_GET_INFO(win, info_used, ierr);
  return ;
}

/******************************************************
***      MPI_Win_get_info wrapper function (lowercase_)
******************************************************/
void mpi_win_get_info_(MPI_Fint * win, MPI_Fint * info_used, MPI_Fint * ierr)
{
  MPI_WIN_GET_INFO(win, info_used, ierr);
  return ;
}

/******************************************************
***      MPI_Win_get_info wrapper function (lowercase__)
******************************************************/
void mpi_win_get_info__(MPI_Fint * win, MPI_Fint * info_used, MPI_Fint * ierr)
{
  MPI_WIN_GET_INFO(win, info_used, ierr);
  return ;
}


/******************************************************
***      MPI_Comm_idup wrapper function 
******************************************************/
int MPI_Comm_idup(MPI_Comm comm, MPI_Comm* newcomm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Comm_idup()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Comm_idup(comm, newcomm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Comm_idup wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMM_IDUP(MPI_Fint * comm, MPI_Fint * newcomm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_Comm local_newcomm;
  MPI_Request local_request;
  *ierr = MPI_Comm_idup(MPI_Comm_f2c(*comm), &local_newcomm, &local_request);
  *newcomm = MPI_Comm_c2f(local_newcomm);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Comm_idup wrapper function (lowercase)
******************************************************/
void mpi_comm_idup(MPI_Fint * comm, MPI_Fint * newcomm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_COMM_IDUP(comm, newcomm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_idup wrapper function (lowercase_)
******************************************************/
void mpi_comm_idup_(MPI_Fint * comm, MPI_Fint * newcomm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_COMM_IDUP(comm, newcomm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_idup wrapper function (lowercase__)
******************************************************/
void mpi_comm_idup__(MPI_Fint * comm, MPI_Fint * newcomm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_COMM_IDUP(comm, newcomm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Comm_create_group wrapper function 
******************************************************/
int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm* newcomm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Comm_create_group()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Comm_create_group(comm, group, tag, newcomm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Comm_create_group wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMM_CREATE_GROUP(MPI_Fint * comm, MPI_Fint * group, MPI_Fint * tag, MPI_Fint * newcomm, MPI_Fint * ierr)
{
   MPI_Comm local_newcomm;
  *ierr = MPI_Comm_create_group(MPI_Comm_f2c(*comm), MPI_Group_f2c(*group),  *tag,  &local_newcomm);
  *newcomm = MPI_Comm_c2f(local_newcomm);
  return ;
}

/******************************************************
***      MPI_Comm_create_group wrapper function (lowercase)
******************************************************/
void mpi_comm_create_group(MPI_Fint * comm, MPI_Fint * group, MPI_Fint * tag, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_GROUP(comm, group, tag, newcomm, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_create_group wrapper function (lowercase_)
******************************************************/
void mpi_comm_create_group_(MPI_Fint * comm, MPI_Fint * group, MPI_Fint * tag, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_GROUP(comm, group, tag, newcomm, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_create_group wrapper function (lowercase__)
******************************************************/
void mpi_comm_create_group__(MPI_Fint * comm, MPI_Fint * group, MPI_Fint * tag, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_GROUP(comm, group, tag, newcomm, ierr);
  return ;
}


/******************************************************
***      MPI_Comm_split_type wrapper function 
******************************************************/
int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm* newcomm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Comm_split_type()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Comm_split_type(comm, split_type, key, info, newcomm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Comm_split_type wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMM_SPLIT_TYPE(MPI_Fint * comm, MPI_Fint * split_type, MPI_Fint * key, MPI_Fint * info, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_Comm local_newcomm;
  *ierr = MPI_Comm_split_type(MPI_Comm_f2c(*comm), *split_type, *key, MPI_Info_f2c(*info), &local_newcomm);
  *newcomm = MPI_Comm_c2f(local_newcomm);
  return ;
}

/******************************************************
***      MPI_Comm_split_type wrapper function (lowercase)
******************************************************/
void mpi_comm_split_type(MPI_Fint * comm, MPI_Fint * split_type, MPI_Fint * key, MPI_Fint * info, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_SPLIT_TYPE(comm, split_type, key, info, newcomm, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_split_type wrapper function (lowercase_)
******************************************************/
void mpi_comm_split_type_(MPI_Fint * comm, MPI_Fint * split_type, MPI_Fint * key, MPI_Fint * info, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_SPLIT_TYPE(comm, split_type, key, info, newcomm, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_split_type wrapper function (lowercase__)
******************************************************/
void mpi_comm_split_type__(MPI_Fint * comm, MPI_Fint * split_type, MPI_Fint * key, MPI_Fint * info, MPI_Fint * newcomm, MPI_Fint * ierr)
{
  MPI_COMM_SPLIT_TYPE(comm, split_type, key, info, newcomm, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_allgather wrapper function 
******************************************************/
int MPI_Neighbor_allgather(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgather()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Neighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_allgather wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_NEIGHBOR_ALLGATHER(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr);

/******************************************************
***      MPI_Neighbor_allgather wrapper function (lowercase)
******************************************************/
void mpi_neighbor_allgather(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_allgather_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_allgather__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_allgatherv wrapper function 
******************************************************/
int MPI_Neighbor_allgatherv(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, TAU_MPICH3_CONST int * recvcounts,
TAU_MPICH3_CONST int * displs, MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgatherv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Neighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_allgatherv wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_NEIGHBOR_ALLGATHERV(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr);

/******************************************************
***      MPI_Neighbor_allgatherv wrapper function (lowercase)
******************************************************/
void mpi_neighbor_allgatherv(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr);
{
  MPI_NEIGHBOR_ALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_allgatherv_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr);
{
  MPI_NEIGHBOR_ALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_allgatherv__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr);
{
  MPI_NEIGHBOR_ALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoall wrapper function 
******************************************************/
int MPI_Neighbor_alltoall(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoall()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Neighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoall wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_NEIGHBOR_ALLTOALL(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr);

/******************************************************
***      MPI_Neighbor_alltoall wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoall(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoall_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoall__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoallv wrapper function 
******************************************************/
int MPI_Neighbor_alltoallv(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int * sendcounts, TAU_MPICH3_CONST int * sdispls, MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST int * recvcounts, TAU_MPICH3_CONST int * rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoallv wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_NEIGHBOR_ALLTOALLV(MPI_Aint * sendbuf, int * sendcounts, int * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
int * recvcounts, int * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr);

/******************************************************
***      MPI_Neighbor_alltoallv wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoallv(MPI_Aint * sendbuf, int * sendcounts, int * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
int * recvcounts, int * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoallv_(MPI_Aint * sendbuf, int * sendcounts, int * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
int * recvcounts, int * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoallv__(MPI_Aint * sendbuf, int * sendcounts, int * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
int * recvcounts, int * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoallw wrapper function 
******************************************************/
int MPI_Neighbor_alltoallw(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int * sendcounts, TAU_MPICH3_CONST MPI_Aint * sdispls,
TAU_MPICH3_CONST MPI_Datatype * sendtypes, void* recvbuf, TAU_MPICH3_CONST int * recvcounts, TAU_MPICH3_CONST MPI_Aint * rdispls,
TAU_MPICH3_CONST MPI_Datatype * recvtypes, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallw()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoallw wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_NEIGHBOR_ALLTOALLW(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtypes,
MPI_Fint * comm, MPI_Fint * ierr);

/******************************************************
***      MPI_Neighbor_alltoallw wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoallw(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtypes,
MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoallw_(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtypes,
MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoallw__(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, MPI_Fint * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtypes,
MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_allgather wrapper function 
******************************************************/
int MPI_Ineighbor_allgather(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_allgather()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_allgather wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_INEIGHBOR_ALLGATHER(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);
/******************************************************
***      MPI_Ineighbor_allgather wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_allgather(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgather wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_allgather_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgather wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_allgather__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request,
    ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_allgatherv wrapper function 
******************************************************/
int MPI_Ineighbor_allgatherv(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, TAU_MPICH3_CONST int * recvcounts,
TAU_MPICH3_CONST int * displs, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_allgatherv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ineighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_allgatherv wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_INEIGHBOR_ALLGATHERV(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ineighbor_allgatherv wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_allgatherv(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgatherv wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_allgatherv_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgatherv wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_allgatherv__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcounts,
MPI_Fint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHERV(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_alltoall wrapper function 
******************************************************/
int MPI_Ineighbor_alltoall(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoall()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ineighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_alltoall wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_INEIGHBOR_ALLTOALL(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ineighbor_alltoall wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_alltoall(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoall wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_alltoall_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoall wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_alltoall__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request,
    ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_alltoallv wrapper function 
******************************************************/
int MPI_Ineighbor_alltoallv(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int * sendcounts, TAU_MPICH3_CONST int * sdispls, MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST int * recvcounts, TAU_MPICH3_CONST int * rdispls, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoallv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_alltoallv wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_INEIGHBOR_ALLTOALLV(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);
/******************************************************
***      MPI_Ineighbor_alltoallv wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_alltoallv(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallv wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_alltoallv_(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallv wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_alltoallv__(MPI_Aint * sendbuf, MPI_Fint * sendcounts, MPI_Fint * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Fint * recvcounts, MPI_Fint * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLV(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_alltoallw wrapper function 
******************************************************/
int MPI_Ineighbor_alltoallw(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int * sendcounts, TAU_MPICH3_CONST MPI_Aint * sdispls,
TAU_MPICH3_CONST MPI_Datatype * sendtypes, void* recvbuf, TAU_MPICH3_CONST int * recvcounts, TAU_MPICH3_CONST MPI_Aint * rdispls,
TAU_MPICH3_CONST MPI_Datatype * recvtypes, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoallw()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Ineighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_alltoallw wrapper function (uppercase Fortran)
******************************************************/
extern void MPI_INEIGHBOR_ALLTOALLW(MPI_Aint * sendbuf, int * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, int * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtypes,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr);

/******************************************************
***      MPI_Ineighbor_alltoallw wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_alltoallw(MPI_Aint * sendbuf, int * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, int * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtypes,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallw wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_alltoallw_(MPI_Aint * sendbuf, int * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, int * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtypes,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallw wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_alltoallw__(MPI_Aint * sendbuf, int * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtypes,
MPI_Aint * recvbuf, int * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtypes,
MPI_Fint * comm, MPI_Fint * request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLW(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request, ierr);
  return ;
}

!!!
/******************************************************
***      MPI_Dist_graph_neighbors wrapper function 
******************************************************/
int MPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int * sources, int * sourceweights, int maxoutdegree,
int * destinations, int * destweights)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Dist_graph_neighbors()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Dist_graph_neighbors(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations,
    destweights);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Dist_graph_neighbors wrapper function (uppercase Fortran)
******************************************************/
void MPI_DIST_GRAPH_NEIGHBORS(MPI_Fint * comm, MPI_Fint * maxindegree, MPI_Fint * sources, MPI_Fint * sourceweights,
MPI_Fint * maxoutdegree, MPI_Fint * destinations, MPI_Fint * destweights, MPI_Fint * ierr)
{

  *ierr = MPI_Dist_graph_neighbors(/* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ maxindegree,
    /* MPI_HANDLE_TYPES */ sources, /* MPI_HANDLE_TYPES */ sourceweights,
    /* MPI_HANDLE_TYPES */ maxoutdegree, /* MPI_HANDLE_TYPES */ destinations,
    /* MPI_HANDLE_TYPES */ destweights);
  return ;
}

/******************************************************
***      MPI_Dist_graph_neighbors wrapper function (lowercase)
******************************************************/
void mpi_dist_graph_neighbors(MPI_Fint * comm, MPI_Fint * maxindegree, MPI_Fint * sources, MPI_Fint * sourceweights,
MPI_Fint * maxoutdegree, MPI_Fint * destinations, MPI_Fint * destweights, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_NEIGHBORS(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations,
    destweights, ierr);
  return ;
}

/******************************************************
***      MPI_Dist_graph_neighbors wrapper function (lowercase_)
******************************************************/
void mpi_dist_graph_neighbors_(MPI_Fint * comm, MPI_Fint * maxindegree, MPI_Fint * sources, MPI_Fint * sourceweights,
MPI_Fint * maxoutdegree, MPI_Fint * destinations, MPI_Fint * destweights, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_NEIGHBORS(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations,
    destweights, ierr);
  return ;
}

/******************************************************
***      MPI_Dist_graph_neighbors wrapper function (lowercase__)
******************************************************/
void mpi_dist_graph_neighbors__(MPI_Fint * comm, MPI_Fint * maxindegree, MPI_Fint * sources, MPI_Fint * sourceweights,
MPI_Fint * maxoutdegree, MPI_Fint * destinations, MPI_Fint * destweights, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_NEIGHBORS(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations,
    destweights, ierr);
  return ;
}


/******************************************************
***      MPI_Dist_graph_create_adjacent wrapper function 
******************************************************/
int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, TAU_MPICH3_CONST int sources[], TAU_MPICH3_CONST int sourceweights[], int outdegree,
TAU_MPICH3_CONST int destinations[], TAU_MPICH3_CONST int destweights[], MPI_Info info, int reorder,
MPI_Comm* comm_dist_graph)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Dist_graph_create_adjacent()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Dist_graph_create_adjacent(comm_old, indegree, sources, sourceweights, outdegree,
    destinations, destweights, info, reorder, comm_dist_graph);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Dist_graph_create_adjacent wrapper function (uppercase Fortran)
******************************************************/
void MPI_DIST_GRAPH_CREATE_ADJACENT(MPI_Fint comm_old, MPI_Fint indegree, int sources[], int sourceweights[],
MPI_Fint outdegree, int destinations[], int destweights[], MPI_Fint info,
MPI_Fint reorder, MPI_Fint comm_dist_graph, MPI_Fint * ierr)
{

  *ierr = MPI_Dist_graph_create_adjacent(/* MPI_HANDLE_TYPES */ comm_old,
    /* MPI_HANDLE_TYPES */ indegree, sources, sourceweights, /* MPI_HANDLE_TYPES */ outdegree,
    destinations, destweights, /* MPI_HANDLE_TYPES */ info, /* MPI_HANDLE_TYPES */ reorder,
    /* MPI_HANDLE_TYPES */ comm_dist_graph);
  return ;
}

/******************************************************
***      MPI_Dist_graph_create_adjacent wrapper function (lowercase)
******************************************************/
void mpi_dist_graph_create_adjacent(MPI_Fint comm_old, MPI_Fint indegree, int sources[], int sourceweights[],
MPI_Fint outdegree, int destinations[], int destweights[], MPI_Fint info,
MPI_Fint reorder, MPI_Fint comm_dist_graph, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_CREATE_ADJACENT(comm_old, indegree, sources, sourceweights, outdegree,
    destinations, destweights, info, reorder, comm_dist_graph, ierr);
  return ;
}

/******************************************************
***      MPI_Dist_graph_create_adjacent wrapper function (lowercase_)
******************************************************/
void mpi_dist_graph_create_adjacent_(MPI_Fint comm_old, MPI_Fint indegree, int sources[], int sourceweights[],
MPI_Fint outdegree, int destinations[], int destweights[], MPI_Fint info,
MPI_Fint reorder, MPI_Fint comm_dist_graph, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_CREATE_ADJACENT(comm_old, indegree, sources, sourceweights, outdegree,
    destinations, destweights, info, reorder, comm_dist_graph, ierr);
  return ;
}

/******************************************************
***      MPI_Dist_graph_create_adjacent wrapper function (lowercase__)
******************************************************/
void mpi_dist_graph_create_adjacent__(MPI_Fint comm_old, MPI_Fint indegree, int sources[], int sourceweights[],
MPI_Fint outdegree, int destinations[], int destweights[], MPI_Fint info,
MPI_Fint reorder, MPI_Fint comm_dist_graph, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_CREATE_ADJACENT(comm_old, indegree, sources, sourceweights, outdegree,
    destinations, destweights, info, reorder, comm_dist_graph, ierr);
  return ;
}


/******************************************************
***      MPI_Rput wrapper function 
******************************************************/
int MPI_Rput(TAU_MPICH3_CONST void* origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Rput()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Rput(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Rput wrapper function (uppercase Fortran)
******************************************************/
void MPI_RPUT(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{

  *ierr = MPI_Rput(origin_addr, /* MPI_HANDLE_TYPES */ origin_count,
    /* MPI_HANDLE_TYPES */ origin_datatype, /* MPI_HANDLE_TYPES */ target_rank, target_disp,
    /* MPI_HANDLE_TYPES */ target_count, /* MPI_HANDLE_TYPES */ target_datatype,
    /* MPI_HANDLE_TYPES */ win, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_Rput wrapper function (lowercase)
******************************************************/
void mpi_rput(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RPUT(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rput wrapper function (lowercase_)
******************************************************/
void mpi_rput_(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RPUT(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rput wrapper function (lowercase__)
******************************************************/
void mpi_rput__(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RPUT(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}


/******************************************************
***      MPI_Rget wrapper function 
******************************************************/
int MPI_Rget(void* origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Rget()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Rget(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Rget wrapper function (uppercase Fortran)
******************************************************/
void MPI_RGET(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{

  *ierr = MPI_Rget(origin_addr, /* MPI_HANDLE_TYPES */ origin_count,
    /* MPI_HANDLE_TYPES */ origin_datatype, /* MPI_HANDLE_TYPES */ target_rank, target_disp,
    /* MPI_HANDLE_TYPES */ target_count, /* MPI_HANDLE_TYPES */ target_datatype,
    /* MPI_HANDLE_TYPES */ win, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_Rget wrapper function (lowercase)
******************************************************/
void mpi_rget(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RGET(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rget wrapper function (lowercase_)
******************************************************/
void mpi_rget_(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RGET(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rget wrapper function (lowercase__)
******************************************************/
void mpi_rget__(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RGET(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}


/******************************************************
***      MPI_Raccumulate wrapper function 
******************************************************/
int MPI_Raccumulate(TAU_MPICH3_CONST void* origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Raccumulate()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Raccumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Raccumulate wrapper function (uppercase Fortran)
******************************************************/
void MPI_RACCUMULATE(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{

  *ierr = MPI_Raccumulate(origin_addr, /* MPI_HANDLE_TYPES */ origin_count,
    /* MPI_HANDLE_TYPES */ origin_datatype, /* MPI_HANDLE_TYPES */ target_rank, target_disp,
    /* MPI_HANDLE_TYPES */ target_count, /* MPI_HANDLE_TYPES */ target_datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ win, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_Raccumulate wrapper function (lowercase)
******************************************************/
void mpi_raccumulate(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RACCUMULATE(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Raccumulate wrapper function (lowercase_)
******************************************************/
void mpi_raccumulate_(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RACCUMULATE(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Raccumulate wrapper function (lowercase__)
******************************************************/
void mpi_raccumulate__(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win,
MPI_Fint request, MPI_Fint * ierr)
{
  MPI_RACCUMULATE(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, request, ierr);
  return ;
}


/******************************************************
***      MPI_Get_accumulate wrapper function 
******************************************************/
int MPI_Get_accumulate(TAU_MPICH3_CONST void* origin_addr, int origin_count, MPI_Datatype origin_datatype, void* result_addr,
int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Get_accumulate()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Get_accumulate(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Get_accumulate wrapper function (uppercase Fortran)
******************************************************/
void MPI_GET_ACCUMULATE(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Fint result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Get_accumulate(origin_addr, /* MPI_HANDLE_TYPES */ origin_count,
    /* MPI_HANDLE_TYPES */ origin_datatype, result_addr, /* MPI_HANDLE_TYPES */ result_count,
    /* MPI_HANDLE_TYPES */ result_datatype, /* MPI_HANDLE_TYPES */ target_rank, target_disp,
    /* MPI_HANDLE_TYPES */ target_count, /* MPI_HANDLE_TYPES */ target_datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Get_accumulate wrapper function (lowercase)
******************************************************/
void mpi_get_accumulate(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Fint result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_GET_ACCUMULATE(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
  return ;
}

/******************************************************
***      MPI_Get_accumulate wrapper function (lowercase_)
******************************************************/
void mpi_get_accumulate_(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Fint result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_GET_ACCUMULATE(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
  return ;
}

/******************************************************
***      MPI_Get_accumulate wrapper function (lowercase__)
******************************************************/
void mpi_get_accumulate__(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Fint result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_GET_ACCUMULATE(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
  return ;
}


/******************************************************
***      MPI_Rget_accumulate wrapper function 
******************************************************/
int MPI_Rget_accumulate(TAU_MPICH3_CONST void* origin_addr, int origin_count, MPI_Datatype origin_datatype, void* result_addr,
int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Rget_accumulate()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Rget_accumulate(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Rget_accumulate wrapper function (uppercase Fortran)
******************************************************/
void MPI_RGET_ACCUMULATE(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Fint result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint request,
MPI_Fint * ierr)
{

  *ierr = MPI_Rget_accumulate(origin_addr, /* MPI_HANDLE_TYPES */ origin_count,
    /* MPI_HANDLE_TYPES */ origin_datatype, result_addr, /* MPI_HANDLE_TYPES */ result_count,
    /* MPI_HANDLE_TYPES */ result_datatype, /* MPI_HANDLE_TYPES */ target_rank, target_disp,
    /* MPI_HANDLE_TYPES */ target_count, /* MPI_HANDLE_TYPES */ target_datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ win, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_Rget_accumulate wrapper function (lowercase)
******************************************************/
void mpi_rget_accumulate(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Fint result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_RGET_ACCUMULATE(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rget_accumulate wrapper function (lowercase_)
******************************************************/
void mpi_rget_accumulate_(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Fint result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_RGET_ACCUMULATE(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rget_accumulate wrapper function (lowercase__)
******************************************************/
void mpi_rget_accumulate__(MPI_Aint origin_addr, MPI_Fint origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Fint result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Fint target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_RGET_ACCUMULATE(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
  return ;
}


/******************************************************
***      MPI_Fetch_and_op wrapper function 
******************************************************/
int MPI_Fetch_and_op(TAU_MPICH3_CONST void* origin_addr, void* result_addr, MPI_Datatype datatype, int target_rank,
MPI_Aint target_disp, MPI_Op op, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Fetch_and_op()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Fetch_and_op(origin_addr, result_addr, datatype, target_rank, target_disp, op, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Fetch_and_op wrapper function (uppercase Fortran)
******************************************************/
void MPI_FETCH_AND_OP(MPI_Aint origin_addr, MPI_Aint result_addr, MPI_Fint datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Fetch_and_op(origin_addr, result_addr, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ target_rank, target_disp, /* MPI_HANDLE_TYPES */ op,
    /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Fetch_and_op wrapper function (lowercase)
******************************************************/
void mpi_fetch_and_op(MPI_Aint origin_addr, MPI_Aint result_addr, MPI_Fint datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_FETCH_AND_OP(origin_addr, result_addr, datatype, target_rank, target_disp, op, win, ierr);
  return ;
}

/******************************************************
***      MPI_Fetch_and_op wrapper function (lowercase_)
******************************************************/
void mpi_fetch_and_op_(MPI_Aint origin_addr, MPI_Aint result_addr, MPI_Fint datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_FETCH_AND_OP(origin_addr, result_addr, datatype, target_rank, target_disp, op, win, ierr);
  return ;
}

/******************************************************
***      MPI_Fetch_and_op wrapper function (lowercase__)
******************************************************/
void mpi_fetch_and_op__(MPI_Aint origin_addr, MPI_Aint result_addr, MPI_Fint datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_FETCH_AND_OP(origin_addr, result_addr, datatype, target_rank, target_disp, op, win, ierr);
  return ;
}


/******************************************************
***      MPI_Compare_and_swap wrapper function 
******************************************************/
int MPI_Compare_and_swap(TAU_MPICH3_CONST void* origin_addr, TAU_MPICH3_CONST void* compare_addr, void* result_addr, MPI_Datatype datatype,
int target_rank, MPI_Aint target_disp, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Compare_and_swap()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp,
    win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Compare_and_swap wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMPARE_AND_SWAP(MPI_Aint origin_addr, MPI_Aint compare_addr, MPI_Aint result_addr, MPI_Fint datatype,
MPI_Fint target_rank, MPI_Aint target_disp, MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Compare_and_swap(origin_addr, compare_addr, result_addr,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ target_rank, target_disp,
    /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Compare_and_swap wrapper function (lowercase)
******************************************************/
void mpi_compare_and_swap(MPI_Aint origin_addr, MPI_Aint compare_addr, MPI_Aint result_addr, MPI_Fint datatype,
MPI_Fint target_rank, MPI_Aint target_disp, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_COMPARE_AND_SWAP(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp,
    win, ierr);
  return ;
}

/******************************************************
***      MPI_Compare_and_swap wrapper function (lowercase_)
******************************************************/
void mpi_compare_and_swap_(MPI_Aint origin_addr, MPI_Aint compare_addr, MPI_Aint result_addr, MPI_Fint datatype,
MPI_Fint target_rank, MPI_Aint target_disp, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_COMPARE_AND_SWAP(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp,
    win, ierr);
  return ;
}

/******************************************************
***      MPI_Compare_and_swap wrapper function (lowercase__)
******************************************************/
void mpi_compare_and_swap__(MPI_Aint origin_addr, MPI_Aint compare_addr, MPI_Aint result_addr, MPI_Fint datatype,
MPI_Fint target_rank, MPI_Aint target_disp, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_COMPARE_AND_SWAP(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp,
    win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_allocate wrapper function 
******************************************************/
int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void* baseptr, MPI_Win* win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_allocate()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_allocate(size, disp_unit, info, comm, baseptr, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_allocate wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_ALLOCATE(MPI_Aint size, MPI_Fint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Fint win,
MPI_Fint * ierr)
{

  *ierr = MPI_Win_allocate(size, /* MPI_HANDLE_TYPES */ disp_unit, /* MPI_HANDLE_TYPES */ info,
    /* MPI_HANDLE_TYPES */ comm, baseptr, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_allocate wrapper function (lowercase)
******************************************************/
void mpi_win_allocate(MPI_Aint size, MPI_Fint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_allocate wrapper function (lowercase_)
******************************************************/
void mpi_win_allocate_(MPI_Aint size, MPI_Fint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_allocate wrapper function (lowercase__)
******************************************************/
void mpi_win_allocate__(MPI_Aint size, MPI_Fint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_allocate_shared wrapper function 
******************************************************/
int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void* baseptr, MPI_Win* win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_allocate_shared()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_allocate_shared(size, disp_unit, info, comm, baseptr, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_allocate_shared wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_ALLOCATE_SHARED(MPI_Aint size, MPI_Fint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Fint win,
MPI_Fint * ierr)
{

  *ierr = MPI_Win_allocate_shared(size, /* MPI_HANDLE_TYPES */ disp_unit,
    /* MPI_HANDLE_TYPES */ info, /* MPI_HANDLE_TYPES */ comm, baseptr, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_allocate_shared wrapper function (lowercase)
******************************************************/
void mpi_win_allocate_shared(MPI_Aint size, MPI_Fint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_SHARED(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_allocate_shared wrapper function (lowercase_)
******************************************************/
void mpi_win_allocate_shared_(MPI_Aint size, MPI_Fint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_SHARED(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_allocate_shared wrapper function (lowercase__)
******************************************************/
void mpi_win_allocate_shared__(MPI_Aint size, MPI_Fint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_SHARED(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_create_dynamic wrapper function 
******************************************************/
int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win* win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_create_dynamic()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_create_dynamic(info, comm, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_create_dynamic wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_CREATE_DYNAMIC(MPI_Fint info, MPI_Fint comm, MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Win_create_dynamic(/* MPI_HANDLE_TYPES */ info, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_create_dynamic wrapper function (lowercase)
******************************************************/
void mpi_win_create_dynamic(MPI_Fint info, MPI_Fint comm, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_DYNAMIC(info, comm, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_create_dynamic wrapper function (lowercase_)
******************************************************/
void mpi_win_create_dynamic_(MPI_Fint info, MPI_Fint comm, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_DYNAMIC(info, comm, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_create_dynamic wrapper function (lowercase__)
******************************************************/
void mpi_win_create_dynamic__(MPI_Fint info, MPI_Fint comm, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_DYNAMIC(info, comm, win, ierr);
  return ;
}


/******************************************************
***      MPI_Dist_graph_create wrapper function 
******************************************************/
int MPI_Dist_graph_create(MPI_Comm comm_old, int n, TAU_MPICH3_CONST int sources[], TAU_MPICH3_CONST int degrees[], TAU_MPICH3_CONST int destinations[],
TAU_MPICH3_CONST int weights[], MPI_Info info, int reorder, MPI_Comm* comm_dist_graph)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Dist_graph_create()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Dist_graph_create(comm_old, n, sources, degrees, destinations, weights, info, reorder,
    comm_dist_graph);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Dist_graph_create wrapper function (uppercase Fortran)
******************************************************/
void MPI_DIST_GRAPH_CREATE(MPI_Fint comm_old, MPI_Fint n, int sources[], int degrees[], int destinations[],
int weights[], MPI_Fint info, MPI_Fint reorder, MPI_Fint comm_dist_graph, MPI_Fint * ierr)
{

  *ierr = MPI_Dist_graph_create(/* MPI_HANDLE_TYPES */ comm_old, /* MPI_HANDLE_TYPES */ n, sources,
    degrees, destinations, weights, /* MPI_HANDLE_TYPES */ info, /* MPI_HANDLE_TYPES */ reorder,
    /* MPI_HANDLE_TYPES */ comm_dist_graph);
  return ;
}

/******************************************************
***      MPI_Dist_graph_create wrapper function (lowercase)
******************************************************/
void mpi_dist_graph_create(MPI_Fint comm_old, MPI_Fint n, int sources[], int degrees[], int destinations[],
int weights[], MPI_Fint info, MPI_Fint reorder, MPI_Fint comm_dist_graph, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_CREATE(comm_old, n, sources, degrees, destinations, weights, info, reorder,
    comm_dist_graph, ierr);
  return ;
}

/******************************************************
***      MPI_Dist_graph_create wrapper function (lowercase_)
******************************************************/
void mpi_dist_graph_create_(MPI_Fint comm_old, MPI_Fint n, int sources[], int degrees[], int destinations[],
int weights[], MPI_Fint info, MPI_Fint reorder, MPI_Fint comm_dist_graph, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_CREATE(comm_old, n, sources, degrees, destinations, weights, info, reorder,
    comm_dist_graph, ierr);
  return ;
}

/******************************************************
***      MPI_Dist_graph_create wrapper function (lowercase__)
******************************************************/
void mpi_dist_graph_create__(MPI_Fint comm_old, MPI_Fint n, int sources[], int degrees[], int destinations[],
int weights[], MPI_Fint info, MPI_Fint reorder, MPI_Fint comm_dist_graph, MPI_Fint * ierr)
{
  MPI_DIST_GRAPH_CREATE(comm_old, n, sources, degrees, destinations, weights, info, reorder,
    comm_dist_graph, ierr);
  return ;
}


/******************************************************
***      MPI_File_iread_all wrapper function 
******************************************************/
int MPI_File_iread_all(MPI_File fh, void* buf, int count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iread_all()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_File_iread_all(fh, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iread_all wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IREAD_ALL(MPI_Fint fh, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iread_all(/* MPI_HANDLE_TYPES */ fh, buf, /* MPI_HANDLE_TYPES */ count,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_File_iread_all wrapper function (lowercase)
******************************************************/
void mpi_file_iread_all(MPI_Fint fh, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_ALL(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_all wrapper function (lowercase_)
******************************************************/
void mpi_file_iread_all_(MPI_Fint fh, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_ALL(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_all wrapper function (lowercase__)
******************************************************/
void mpi_file_iread_all__(MPI_Fint fh, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_ALL(fh, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iread_at_all wrapper function 
******************************************************/
int MPI_File_iread_at_all(MPI_File fh, MPI_Offset offset, void* buf, int count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iread_at_all()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_File_iread_at_all(fh, offset, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iread_at_all wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IREAD_AT_ALL(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request,
MPI_Fint * ierr)
{

  *ierr = MPI_File_iread_at_all(/* MPI_HANDLE_TYPES */ fh, offset, buf,
    /* MPI_HANDLE_TYPES */ count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_File_iread_at_all wrapper function (lowercase)
******************************************************/
void mpi_file_iread_at_all(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_ALL(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_at_all wrapper function (lowercase_)
******************************************************/
void mpi_file_iread_at_all_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_ALL(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_at_all wrapper function (lowercase__)
******************************************************/
void mpi_file_iread_at_all__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_ALL(fh, offset, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iwrite_at_all wrapper function 
******************************************************/
int MPI_File_iwrite_at_all(MPI_File fh, MPI_Offset offset, TAU_MPICH3_CONST void* buf, int count, MPI_Datatype datatype,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_at_all()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_File_iwrite_at_all(fh, offset, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iwrite_at_all wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IWRITE_AT_ALL(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request,
MPI_Fint * ierr)
{

  *ierr = MPI_File_iwrite_at_all(/* MPI_HANDLE_TYPES */ fh, offset, buf,
    /* MPI_HANDLE_TYPES */ count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_all wrapper function (lowercase)
******************************************************/
void mpi_file_iwrite_at_all(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_ALL(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_all wrapper function (lowercase_)
******************************************************/
void mpi_file_iwrite_at_all_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_ALL(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_all wrapper function (lowercase__)
******************************************************/
void mpi_file_iwrite_at_all__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Fint count, MPI_Fint datatype, MPI_Fint request,
MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_ALL(fh, offset, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_Open_port wrapper function 
******************************************************/
int MPI_Open_port(MPI_Info info, char* port_name)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Open_port()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Open_port(info, port_name);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Open_port wrapper function (uppercase Fortran)
******************************************************/
void MPI_OPEN_PORT(MPI_Fint info, char* port_name, MPI_Fint * ierr)
{

  *ierr = MPI_Open_port(/* MPI_HANDLE_TYPES */ info, port_name);
  return ;
}

/******************************************************
***      MPI_Open_port wrapper function (lowercase)
******************************************************/
void mpi_open_port(MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_OPEN_PORT(info, port_name, ierr);
  return ;
}

/******************************************************
***      MPI_Open_port wrapper function (lowercase_)
******************************************************/
void mpi_open_port_(MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_OPEN_PORT(info, port_name, ierr);
  return ;
}

/******************************************************
***      MPI_Open_port wrapper function (lowercase__)
******************************************************/
void mpi_open_port__(MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_OPEN_PORT(info, port_name, ierr);
  return ;
}


/******************************************************
***      MPI_Publish_name wrapper function 
******************************************************/
int MPI_Publish_name(TAU_MPICH3_CONST char* service_name, MPI_Info info, TAU_MPICH3_CONST char* port_name)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Publish_name()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Publish_name(service_name, info, port_name);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Publish_name wrapper function (uppercase Fortran)
******************************************************/
void MPI_PUBLISH_NAME(char* service_name, MPI_Fint info, char* port_name, MPI_Fint * ierr)
{

  *ierr = MPI_Publish_name(service_name, /* MPI_HANDLE_TYPES */ info, port_name);
  return ;
}

/******************************************************
***      MPI_Publish_name wrapper function (lowercase)
******************************************************/
void mpi_publish_name(char* service_name, MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_PUBLISH_NAME(service_name, info, port_name, ierr);
  return ;
}

/******************************************************
***      MPI_Publish_name wrapper function (lowercase_)
******************************************************/
void mpi_publish_name_(char* service_name, MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_PUBLISH_NAME(service_name, info, port_name, ierr);
  return ;
}

/******************************************************
***      MPI_Publish_name wrapper function (lowercase__)
******************************************************/
void mpi_publish_name__(char* service_name, MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_PUBLISH_NAME(service_name, info, port_name, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_local wrapper function 
******************************************************/
int MPI_Reduce_local(TAU_MPICH3_CONST void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_local()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Reduce_local(inbuf, inoutbuf, count, datatype, op);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_local wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_LOCAL(MPI_Aint inbuf, MPI_Aint inoutbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_local(inbuf, inoutbuf, /* MPI_HANDLE_TYPES */ count,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ op);
  return ;
}

/******************************************************
***      MPI_Reduce_local wrapper function (lowercase)
******************************************************/
void mpi_reduce_local(MPI_Aint inbuf, MPI_Aint inoutbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint * ierr)
{
  MPI_REDUCE_LOCAL(inbuf, inoutbuf, count, datatype, op, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_local wrapper function (lowercase_)
******************************************************/
void mpi_reduce_local_(MPI_Aint inbuf, MPI_Aint inoutbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint * ierr)
{
  MPI_REDUCE_LOCAL(inbuf, inoutbuf, count, datatype, op, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_local wrapper function (lowercase__)
******************************************************/
void mpi_reduce_local__(MPI_Aint inbuf, MPI_Aint inoutbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint * ierr)
{
  MPI_REDUCE_LOCAL(inbuf, inoutbuf, count, datatype, op, ierr);
  return ;
}


/******************************************************
***      MPI_Unpublish_name wrapper function 
******************************************************/
int MPI_Unpublish_name(TAU_MPICH3_CONST char* service_name, MPI_Info info, TAU_MPICH3_CONST char* port_name)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Unpublish_name()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Unpublish_name(service_name, info, port_name);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Unpublish_name wrapper function (uppercase Fortran)
******************************************************/
void MPI_UNPUBLISH_NAME(char* service_name, MPI_Fint info, char* port_name, MPI_Fint * ierr)
{

  *ierr = MPI_Unpublish_name(service_name, /* MPI_HANDLE_TYPES */ info, port_name);
  return ;
}

/******************************************************
***      MPI_Unpublish_name wrapper function (lowercase)
******************************************************/
void mpi_unpublish_name(char* service_name, MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_UNPUBLISH_NAME(service_name, info, port_name, ierr);
  return ;
}

/******************************************************
***      MPI_Unpublish_name wrapper function (lowercase_)
******************************************************/
void mpi_unpublish_name_(char* service_name, MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_UNPUBLISH_NAME(service_name, info, port_name, ierr);
  return ;
}

/******************************************************
***      MPI_Unpublish_name wrapper function (lowercase__)
******************************************************/
void mpi_unpublish_name__(char* service_name, MPI_Fint info, char* port_name, MPI_Fint * ierr)
{
  MPI_UNPUBLISH_NAME(service_name, info, port_name, ierr);
  return ;
}


/******************************************************
***      MPI_Win_attach wrapper function 
******************************************************/
int MPI_Win_attach(MPI_Win win, void* base, MPI_Aint size)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_attach()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_attach(win, base, size);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_attach wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_ATTACH(MPI_Fint win, MPI_Aint base, MPI_Aint size, MPI_Fint * ierr)
{

  *ierr = MPI_Win_attach(/* MPI_HANDLE_TYPES */ win, base, size);
  return ;
}

/******************************************************
***      MPI_Win_attach wrapper function (lowercase)
******************************************************/
void mpi_win_attach(MPI_Fint win, MPI_Aint base, MPI_Aint size, MPI_Fint * ierr)
{
  MPI_WIN_ATTACH(win, base, size, ierr);
  return ;
}

/******************************************************
***      MPI_Win_attach wrapper function (lowercase_)
******************************************************/
void mpi_win_attach_(MPI_Fint win, MPI_Aint base, MPI_Aint size, MPI_Fint * ierr)
{
  MPI_WIN_ATTACH(win, base, size, ierr);
  return ;
}

/******************************************************
***      MPI_Win_attach wrapper function (lowercase__)
******************************************************/
void mpi_win_attach__(MPI_Fint win, MPI_Aint base, MPI_Aint size, MPI_Fint * ierr)
{
  MPI_WIN_ATTACH(win, base, size, ierr);
  return ;
}


/******************************************************
***      MPI_Win_detach wrapper function 
******************************************************/
int MPI_Win_detach(MPI_Win win, TAU_MPICH3_CONST void* base)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_detach()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_detach(win, base);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_detach wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_DETACH(MPI_Fint win, MPI_Aint base, MPI_Fint * ierr)
{

  *ierr = MPI_Win_detach(/* MPI_HANDLE_TYPES */ win, base);
  return ;
}

/******************************************************
***      MPI_Win_detach wrapper function (lowercase)
******************************************************/
void mpi_win_detach(MPI_Fint win, MPI_Aint base, MPI_Fint * ierr)
{
  MPI_WIN_DETACH(win, base, ierr);
  return ;
}

/******************************************************
***      MPI_Win_detach wrapper function (lowercase_)
******************************************************/
void mpi_win_detach_(MPI_Fint win, MPI_Aint base, MPI_Fint * ierr)
{
  MPI_WIN_DETACH(win, base, ierr);
  return ;
}

/******************************************************
***      MPI_Win_detach wrapper function (lowercase__)
******************************************************/
void mpi_win_detach__(MPI_Fint win, MPI_Aint base, MPI_Fint * ierr)
{
  MPI_WIN_DETACH(win, base, ierr);
  return ;
}


/******************************************************
***      MPI_Win_flush wrapper function 
******************************************************/
int MPI_Win_flush(int rank, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_flush()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_flush(rank, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_flush wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_FLUSH(MPI_Fint rank, MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Win_flush(/* MPI_HANDLE_TYPES */ rank, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_flush wrapper function (lowercase)
******************************************************/
void mpi_win_flush(MPI_Fint rank, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH(rank, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_flush wrapper function (lowercase_)
******************************************************/
void mpi_win_flush_(MPI_Fint rank, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH(rank, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_flush wrapper function (lowercase__)
******************************************************/
void mpi_win_flush__(MPI_Fint rank, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH(rank, win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_flush_all wrapper function 
******************************************************/
int MPI_Win_flush_all(MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_flush_all()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_flush_all(win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_flush_all wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_FLUSH_ALL(MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Win_flush_all(/* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_flush_all wrapper function (lowercase)
******************************************************/
void mpi_win_flush_all(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_ALL(win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_flush_all wrapper function (lowercase_)
******************************************************/
void mpi_win_flush_all_(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_ALL(win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_flush_all wrapper function (lowercase__)
******************************************************/
void mpi_win_flush_all__(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_ALL(win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_flush_local wrapper function 
******************************************************/
int MPI_Win_flush_local(int rank, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_flush_local()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_flush_local(rank, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_flush_local wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_FLUSH_LOCAL(MPI_Fint rank, MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Win_flush_local(/* MPI_HANDLE_TYPES */ rank, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_flush_local wrapper function (lowercase)
******************************************************/
void mpi_win_flush_local(MPI_Fint rank, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_LOCAL(rank, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_flush_local wrapper function (lowercase_)
******************************************************/
void mpi_win_flush_local_(MPI_Fint rank, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_LOCAL(rank, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_flush_local wrapper function (lowercase__)
******************************************************/
void mpi_win_flush_local__(MPI_Fint rank, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_LOCAL(rank, win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_flush_local_all wrapper function 
******************************************************/
int MPI_Win_flush_local_all(MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_flush_local_all()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_flush_local_all(win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_flush_local_all wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_FLUSH_LOCAL_ALL(MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Win_flush_local_all(/* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_flush_local_all wrapper function (lowercase)
******************************************************/
void mpi_win_flush_local_all(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_LOCAL_ALL(win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_flush_local_all wrapper function (lowercase_)
******************************************************/
void mpi_win_flush_local_all_(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_LOCAL_ALL(win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_flush_local_all wrapper function (lowercase__)
******************************************************/
void mpi_win_flush_local_all__(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_FLUSH_LOCAL_ALL(win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_lock_all wrapper function 
******************************************************/
int MPI_Win_lock_all(int assert, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_lock_all()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_lock_all(assert, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_lock_all wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_LOCK_ALL(MPI_Fint assert, MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Win_lock_all(/* MPI_HANDLE_TYPES */ assert, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_lock_all wrapper function (lowercase)
******************************************************/
void mpi_win_lock_all(MPI_Fint assert, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_LOCK_ALL(assert, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_lock_all wrapper function (lowercase_)
******************************************************/
void mpi_win_lock_all_(MPI_Fint assert, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_LOCK_ALL(assert, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_lock_all wrapper function (lowercase__)
******************************************************/
void mpi_win_lock_all__(MPI_Fint assert, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_LOCK_ALL(assert, win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_shared_query wrapper function 
******************************************************/
int MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint* size, int* disp_unit, void* baseptr)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_shared_query()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_shared_query(win, rank, size, disp_unit, baseptr);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_shared_query wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_SHARED_QUERY(MPI_Fint win, MPI_Fint rank, MPI_Aint* size, int* disp_unit, MPI_Aint baseptr, MPI_Fint * ierr)
{

  *ierr = MPI_Win_shared_query(/* MPI_HANDLE_TYPES */ win, /* MPI_HANDLE_TYPES */ rank, size,
    disp_unit, baseptr);
  return ;
}

/******************************************************
***      MPI_Win_shared_query wrapper function (lowercase)
******************************************************/
void mpi_win_shared_query(MPI_Fint win, MPI_Fint rank, MPI_Aint* size, int* disp_unit, MPI_Aint baseptr, MPI_Fint * ierr)
{
  MPI_WIN_SHARED_QUERY(win, rank, size, disp_unit, baseptr, ierr);
  return ;
}

/******************************************************
***      MPI_Win_shared_query wrapper function (lowercase_)
******************************************************/
void mpi_win_shared_query_(MPI_Fint win, MPI_Fint rank, MPI_Aint* size, int* disp_unit, MPI_Aint baseptr, MPI_Fint * ierr)
{
  MPI_WIN_SHARED_QUERY(win, rank, size, disp_unit, baseptr, ierr);
  return ;
}

/******************************************************
***      MPI_Win_shared_query wrapper function (lowercase__)
******************************************************/
void mpi_win_shared_query__(MPI_Fint win, MPI_Fint rank, MPI_Aint* size, int* disp_unit, MPI_Aint baseptr, MPI_Fint * ierr)
{
  MPI_WIN_SHARED_QUERY(win, rank, size, disp_unit, baseptr, ierr);
  return ;
}


/******************************************************
***      MPI_Win_sync wrapper function 
******************************************************/
int MPI_Win_sync(MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_sync()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_sync(win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_sync wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_SYNC(MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Win_sync(/* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_sync wrapper function (lowercase)
******************************************************/
void mpi_win_sync(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_SYNC(win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_sync wrapper function (lowercase_)
******************************************************/
void mpi_win_sync_(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_SYNC(win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_sync wrapper function (lowercase__)
******************************************************/
void mpi_win_sync__(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_SYNC(win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_unlock_all wrapper function 
******************************************************/
int MPI_Win_unlock_all(MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_unlock_all()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  PMPI_Win_unlock_all(win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_unlock_all wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_UNLOCK_ALL(MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Win_unlock_all(/* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Win_unlock_all wrapper function (lowercase)
******************************************************/
void mpi_win_unlock_all(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_UNLOCK_ALL(win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_unlock_all wrapper function (lowercase_)
******************************************************/
void mpi_win_unlock_all_(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_UNLOCK_ALL(win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_unlock_all wrapper function (lowercase__)
******************************************************/
void mpi_win_unlock_all__(MPI_Fint win, MPI_Fint * ierr)
{
  MPI_WIN_UNLOCK_ALL(win, ierr);
  return ;
}


