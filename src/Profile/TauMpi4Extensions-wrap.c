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

//For functions where a parameter is [] and needs type transformation, follow
// https://github.com/UO-OACISS/tau2/blob/8eee7ef82b5aec420ccf49848756bb0a8ad251c4/src/Profile/TauFMpi.c#L7378
//https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Isendrecv.3.html
//https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf
//https://www.mpich.org/about/news/
//https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node606.htm
//https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf
//https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Iscatterv.3.html
//https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf
//https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node606.htm


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

static int sum_array_w (TAU_MPICH3_CONST int *counts, TAU_MPICH3_CONST  MPI_Datatype *type, MPI_Comm comm) {

  int typesize, commSize, commRank, i = 0;
  int total = 0;
  PMPI_Comm_rank(comm, &commRank);
  PMPI_Comm_size(comm, &commSize);

  for (i = 0; i<commSize; i++)
  {
    if (type[i] != MPI_DATATYPE_NULL) {
      PMPI_Type_size(type[i], &typesize );
      total += (counts[i] * typesize); // sum
    }
  }

  return total ;
}

/******************************************************
***      MPI_Isendrecv wrapper function 
******************************************************/
int MPI_Isendrecv(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void* recvbuf,
int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Isendrecv()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Isendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Isendrecv wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISENDRECV(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Fint * dest, MPI_Fint * sendtag,
MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * recvtype, MPI_Fint * source, MPI_Fint * recvtag,
MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Isendrecv(sendbuf, *sendcount, MPI_Type_f2c(*sendtype),
    *dest, *sendtag, recvbuf, *recvcount, MPI_Type_f2c(*recvtype),
    *source, *recvtag, MPI_Comm_f2c(*comm), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Isendrecv wrapper function (lowercase)
******************************************************/
void mpi_isendrecv(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Fint * dest, MPI_Fint * sendtag,
MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * recvtype, MPI_Fint * source, MPI_Fint * recvtag,
MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ISENDRECV(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isendrecv wrapper function (lowercase_)
******************************************************/
void mpi_isendrecv_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Fint * dest, MPI_Fint * sendtag,
MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * recvtype, MPI_Fint * source, MPI_Fint * recvtag,
MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ISENDRECV(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isendrecv wrapper function (lowercase__)
******************************************************/
void mpi_isendrecv__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Fint * dest, MPI_Fint * sendtag,
MPI_Aint * recvbuf, MPI_Fint * recvcount, MPI_Fint * recvtype, MPI_Fint * source, MPI_Fint * recvtag,
MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ISENDRECV(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Isendrecv_replace wrapper function 
******************************************************/
int MPI_Isendrecv_replace(void * buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag,
MPI_Comm comm, MPI_Request * request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Isendrecv_replace()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Isendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Isendrecv_replace wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISENDRECV_REPLACE(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * dest, MPI_Fint * sendtag, 
MPI_Fint * source, MPI_Fint * recvtag, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Isendrecv_replace(buf, *count, MPI_Type_f2c(*datatype),
   *dest,*sendtag, *source, *recvtag, MPI_Comm_f2c(*comm), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Isendrecv_replace wrapper function (lowercase)
******************************************************/
void mpi_isendrecv_replace(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * dest, MPI_Fint * sendtag, 
MPI_Fint * source, MPI_Fint * recvtag, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_REPLACE(buf, count, datatype, dest, sendtag, source, recvtag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isendrecv_replace wrapper function (lowercase_)
******************************************************/
void mpi_isendrecv_replace_(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * dest, MPI_Fint * sendtag, 
MPI_Fint * source, MPI_Fint * recvtag, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_REPLACE(buf, count, datatype, dest, sendtag, source, recvtag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isendrecv_replace wrapper function (lowercase__)
******************************************************/
void mpi_isendrecv_replace__(MPI_Aint * buf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * dest, MPI_Fint * sendtag, 
MPI_Fint * source, MPI_Fint * recvtag, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_REPLACE(buf, count, datatype, dest, sendtag, source, recvtag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Comm_idup_with_info wrapper function 
******************************************************/
int MPI_Comm_idup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm* newcomm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Comm_idup_with_info()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Comm_idup_with_info(comm, info, newcomm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Comm_idup_with_info wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMM_IDUP_WITH_INFO(MPI_Fint * comm, MPI_Fint * info, MPI_Comm * newcomm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  MPI_Comm local_newcomm;
  *ierr = MPI_Comm_idup_with_info( MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_newcomm, &local_request);
  *request = MPI_Request_c2f(local_request);
  *newcomm = MPI_Comm_c2f(local_newcomm);
  return ;
}

/******************************************************
***      MPI_Comm_idup_with_info wrapper function (lowercase)
******************************************************/
void mpi_comm_idup_with_info(MPI_Fint * comm, MPI_Fint * info, MPI_Comm * newcomm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_COMM_IDUP_WITH_INFO(comm, info, newcomm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_idup_with_info wrapper function (lowercase_)
******************************************************/
void mpi_comm_idup_with_info_(MPI_Fint * comm, MPI_Fint * info, MPI_Comm * newcomm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_COMM_IDUP_WITH_INFO(comm, info, newcomm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_idup_with_info wrapper function (lowercase__)
******************************************************/
void mpi_comm_idup_with_info__(MPI_Fint * comm, MPI_Fint * info, MPI_Comm * newcomm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_COMM_IDUP_WITH_INFO(comm, info, newcomm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Info_get_string wrapper function 
******************************************************/
int MPI_Info_get_string(MPI_Info info, TAU_MPICH3_CONST char* key, int* buflen, char* value, int* flag)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Info_get_string()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Info_get_string(info, key, buflen, value, flag);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Info_get_string wrapper function (uppercase Fortran)
******************************************************/
void MPI_INFO_GET_STRING(MPI_Fint * info, char * key, int * buflen, char * value, int * flag, MPI_Fint * ierr)
{

  *ierr = MPI_Info_get_string( MPI_Info_f2c(*info), key, buflen, value, flag);
  return ;
}

/******************************************************
***      MPI_Info_get_string wrapper function (lowercase)
******************************************************/
void mpi_info_get_string(MPI_Fint * info, char * key, int * buflen, char * value, int * flag, MPI_Fint * ierr)
{
  MPI_INFO_GET_STRING(info, key, buflen, value, flag, ierr);
  return ;
}

/******************************************************
***      MPI_Info_get_string wrapper function (lowercase_)
******************************************************/
void mpi_info_get_string_(MPI_Fint * info, char * key, int * buflen, char * value, int * flag, MPI_Fint * ierr)
{
  MPI_INFO_GET_STRING(info, key, buflen, value, flag, ierr);
  return ;
}

/******************************************************
***      MPI_Info_get_string wrapper function (lowercase__)
******************************************************/
void mpi_info_get_string__(MPI_Fint * info, char * key, int * buflen, char * value, int * flag, MPI_Fint * ierr)
{
  MPI_INFO_GET_STRING(info, key, buflen, value, flag, ierr);
  return ;
}


/******************************************************
***      MPI_Info_create_env wrapper function 
******************************************************/
int MPI_Info_create_env(int argc, char **argv, MPI_Info* info)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Info_create_env()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Info_create_env(argc, argv, info);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Info_create_env wrapper function (uppercase Fortran)
******************************************************/
void MPI_INFO_CREATE_ENV(MPI_Fint * argc, char **argv, MPI_Info * info, MPI_Fint * ierr)
{
  MPI_Info local_info; 
  *ierr = MPI_Info_create_env( *argc, argv, &local_info) ; 
  *info = MPI_Info_c2f(local_info);
  return ;
}

/******************************************************
***      MPI_Info_create_env wrapper function (lowercase)
******************************************************/
void mpi_info_create_env(MPI_Fint * argc, char **argv, MPI_Info * info, MPI_Fint * ierr)
{
  MPI_INFO_CREATE_ENV(argc, argv, info, ierr);
  return ;
}

/******************************************************
***      MPI_Info_create_env wrapper function (lowercase_)
******************************************************/
void mpi_info_create_env_(MPI_Fint * argc, char **argv, MPI_Info * info, MPI_Fint * ierr)
{
  MPI_INFO_CREATE_ENV(argc, argv, info, ierr);
  return ;
}

/******************************************************
***      MPI_Info_create_env wrapper function (lowercase__)
******************************************************/
void mpi_info_create_env__(MPI_Fint * argc, char **argv, MPI_Info * info, MPI_Fint * ierr)
{
  MPI_INFO_CREATE_ENV(argc, argv, info, ierr);
  return ;
}


//Large count functions and _init
/******************************************************
***      MPI_Accumulate_c wrapper function 
******************************************************/
int MPI_Accumulate_c(TAU_MPICH3_CONST void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Accumulate_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Accumulate_c(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Accumulate_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ACCUMULATE_C(MPI_Aint * origin_addr, MPI_Count * origin_count, MPI_Fint * origin_datatype, MPI_Fint * target_rank,
MPI_Aint * target_disp, MPI_Count * target_count, MPI_Fint * target_datatype, MPI_Fint * op, MPI_Fint * win, MPI_Fint * ierr)
{

  *ierr = MPI_Accumulate_c(origin_addr, *origin_count, MPI_Type_f2c(*origin_datatype),
    *target_rank, *target_disp, *target_count, MPI_Type_f2c(*target_datatype), MPI_Op_f2c(*op), MPI_Win_f2c(*win));
  return ;
}

/******************************************************
***      MPI_Accumulate_c wrapper function (lowercase)
******************************************************/
void mpi_accumulate_c(MPI_Aint * origin_addr, MPI_Count * origin_count, MPI_Fint * origin_datatype, MPI_Fint * target_rank,
MPI_Aint * target_disp, MPI_Count * target_count, MPI_Fint * target_datatype, MPI_Fint * op, MPI_Fint * win, MPI_Fint * ierr)
{
  MPI_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, ierr);
  return ;
}

/******************************************************
***      MPI_Accumulate_c wrapper function (lowercase_)
******************************************************/
void mpi_accumulate_c_(MPI_Aint * origin_addr, MPI_Count * origin_count, MPI_Fint * origin_datatype, MPI_Fint * target_rank,
MPI_Aint * target_disp, MPI_Count * target_count, MPI_Fint * target_datatype, MPI_Fint * op, MPI_Fint * win, MPI_Fint * ierr)
{
  MPI_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, ierr);
  return ;
}

/******************************************************
***      MPI_Accumulate_c wrapper function (lowercase__)
******************************************************/
void mpi_accumulate_c__(MPI_Aint * origin_addr, MPI_Count * origin_count, MPI_Fint * origin_datatype, MPI_Fint * target_rank,
MPI_Aint * target_disp, MPI_Count * target_count, MPI_Fint * target_datatype, MPI_Fint * op, MPI_Fint * win, MPI_Fint * ierr)
{
  MPI_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, ierr);
  return ;
}


/******************************************************
***      MPI_Raccumulate_c wrapper function 
******************************************************/
int MPI_Raccumulate_c(TAU_MPICH3_CONST void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Raccumulate_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Raccumulate_c(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Raccumulate_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_RACCUMULATE_C(MPI_Aint * origin_addr, MPI_Count * origin_count, MPI_Fint * origin_datatype, MPI_Fint * target_rank,
MPI_Aint * target_disp, MPI_Count * target_count, MPI_Fint * target_datatype, MPI_Fint * op, MPI_Fint * win,
MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Raccumulate_c(origin_addr, *origin_count, MPI_Type_f2c(*origin_datatype),
    *target_rank, *target_disp, *target_count, MPI_Type_f2c(*target_datatype), MPI_Op_f2c(*op), MPI_Win_f2c(*win),
    &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Raccumulate_c wrapper function (lowercase)
******************************************************/
void mpi_raccumulate_c(MPI_Aint * origin_addr, MPI_Count * origin_count, MPI_Fint * origin_datatype, MPI_Fint * target_rank,
MPI_Aint * target_disp, MPI_Count * target_count, MPI_Fint * target_datatype, MPI_Fint * op, MPI_Fint * win,
MPI_Request * request, MPI_Fint * ierr)
{
  MPI_RACCUMULATE_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Raccumulate_c wrapper function (lowercase_)
******************************************************/
void mpi_raccumulate_c_(MPI_Aint * origin_addr, MPI_Count * origin_count, MPI_Fint * origin_datatype, MPI_Fint * target_rank,
MPI_Aint * target_disp, MPI_Count * target_count, MPI_Fint * target_datatype, MPI_Fint * op, MPI_Fint * win,
MPI_Request * request, MPI_Fint * ierr)
{
  MPI_RACCUMULATE_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Raccumulate_c wrapper function (lowercase__)
******************************************************/
void mpi_raccumulate_c__(MPI_Aint * origin_addr, MPI_Count * origin_count, MPI_Fint * origin_datatype, MPI_Fint * target_rank,
MPI_Aint * target_disp, MPI_Count * target_count, MPI_Fint * target_datatype, MPI_Fint * op, MPI_Fint * win,
MPI_Request * request, MPI_Fint * ierr)
{
  MPI_RACCUMULATE_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp,
    target_count, target_datatype, op, win, request, ierr);
  return ;
}


/******************************************************
***      MPI_Allgather_c wrapper function 
******************************************************/
int MPI_Allgather_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allgather_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allgather_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allgather_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLGATHER_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{

  *ierr = MPI_Allgather_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, *recvcount,
    MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm));
  return ;
}

/******************************************************
***      MPI_Allgather_c wrapper function (lowercase)
******************************************************/
void mpi_allgather_c(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Allgather_c wrapper function (lowercase_)
******************************************************/
void mpi_allgather_c_(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Allgather_c wrapper function (lowercase__)
******************************************************/
void mpi_allgather_c__(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Iallgather_c wrapper function 
******************************************************/
int MPI_Iallgather_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iallgather_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Iallgather_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iallgather_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IALLGATHER_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Iallgather_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, *recvcount,
     MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), &local_request) ; 
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Iallgather_c wrapper function (lowercase)
******************************************************/
void mpi_iallgather_c(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallgather_c wrapper function (lowercase_)
******************************************************/
void mpi_iallgather_c_(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallgather_c wrapper function (lowercase__)
******************************************************/
void mpi_iallgather_c__(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Allgather_init wrapper function 
******************************************************/
int MPI_Allgather_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allgather_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allgather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allgather_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLGATHER_INIT(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Allgather_init(sendbuf, *sendcount,  MPI_Type_f2c(*sendtype), recvbuf, *recvcount,
    MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_request) ; 
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Allgather_init wrapper function (lowercase)
******************************************************/
void mpi_allgather_init(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Allgather_init wrapper function (lowercase_)
******************************************************/
void mpi_allgather_init_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Allgather_init wrapper function (lowercase__)
******************************************************/
void mpi_allgather_init__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Allgather_init_c wrapper function 
******************************************************/
int MPI_Allgather_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allgather_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allgather_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allgather_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLGATHER_INIT_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Allgather_init_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf,
    *recvcount, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), &local_request) ; 
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Allgather_init_c wrapper function (lowercase)
******************************************************/
void mpi_allgather_init_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLGATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Allgather_init_c wrapper function (lowercase_)
******************************************************/
void mpi_allgather_init_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLGATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Allgather_init_c wrapper function (lowercase__)
******************************************************/
void mpi_allgather_init_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLGATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Allgatherv_c wrapper function 
******************************************************/
int MPI_Allgatherv_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count *recvcounts, TAU_MPICH3_CONST MPI_Aint *displs, MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allgatherv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allgatherv_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allgatherv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLGATHERV_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Allgatherv_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, recvcounts,
    displs, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm));
  return ;
}

/******************************************************
***      MPI_Allgatherv_c wrapper function (lowercase)
******************************************************/
void mpi_allgatherv_c(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Allgatherv_c wrapper function (lowercase_)
******************************************************/
void mpi_allgatherv_c_(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Allgatherv_c wrapper function (lowercase__)
******************************************************/
void mpi_allgatherv_c__(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Iallgatherv_c wrapper function 
******************************************************/
int MPI_Iallgatherv_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count *recvcounts, TAU_MPICH3_CONST MPI_Aint *displs, MPI_Datatype recvtype, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iallgatherv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Iallgatherv_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iallgatherv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IALLGATHERV_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Iallgatherv_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf,
    recvcounts, displs, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Iallgatherv_c wrapper function (lowercase)
******************************************************/
void mpi_iallgatherv_c(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallgatherv_c wrapper function (lowercase_)
******************************************************/
void mpi_iallgatherv_c_(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallgatherv_c wrapper function (lowercase__)
******************************************************/
void mpi_iallgatherv_c__(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Allgatherv_init wrapper function 
******************************************************/
int MPI_Allgatherv_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, TAU_MPICH3_CONST int *recvcounts,
TAU_MPICH3_CONST int *displs, MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allgatherv_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allgatherv_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allgatherv_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLGATHERV_INIT(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, int * recvcounts,
int * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Allgatherv_init(sendbuf, *sendcount,  MPI_Type_f2c(*sendtype), recvbuf, recvcounts, 
    displs, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Allgatherv_init wrapper function (lowercase)
******************************************************/
void mpi_allgatherv_init(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, int * recvcounts,
int * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Allgatherv_init wrapper function (lowercase_)
******************************************************/
void mpi_allgatherv_init_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, int * recvcounts,
int * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Allgatherv_init wrapper function (lowercase__)
******************************************************/
void mpi_allgatherv_init__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, int * recvcounts,
int * displs, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Allgatherv_init_c wrapper function 
******************************************************/
int MPI_Allgatherv_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count *recvcounts, TAU_MPICH3_CONST MPI_Aint *displs, MPI_Datatype recvtype, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allgatherv_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allgatherv_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allgatherv_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLGATHERV_INIT_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Allgatherv_init_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf,
    recvcounts, displs, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Allgatherv_init_c wrapper function (lowercase)
******************************************************/
void mpi_allgatherv_init_c(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Allgatherv_init_c wrapper function (lowercase_)
******************************************************/
void mpi_allgatherv_init_c_(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Allgatherv_init_c wrapper function (lowercase__)
******************************************************/
void mpi_allgatherv_init_c__(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf,
MPI_Count * recvcounts, MPI_Aint * displs, MPI_Fint * recvtype, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLGATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
    info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Allreduce_c wrapper function 
******************************************************/
int MPI_Allreduce_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allreduce_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allreduce_c(sendbuf, recvbuf, count, datatype, op, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allreduce_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLREDUCE_C(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Allreduce_c(sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), MPI_Comm_f2c(*comm));
  return ;
}

/******************************************************
***      MPI_Allreduce_c wrapper function (lowercase)
******************************************************/
void mpi_allreduce_c(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * ierr)
{
  MPI_ALLREDUCE_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Allreduce_c wrapper function (lowercase_)
******************************************************/
void mpi_allreduce_c_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * ierr)
{
  MPI_ALLREDUCE_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Allreduce_c wrapper function (lowercase__)
******************************************************/
void mpi_allreduce_c__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * ierr)
{
  MPI_ALLREDUCE_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Iallreduce_c wrapper function 
******************************************************/
int MPI_Iallreduce_c(TAU_MPICH3_CONST void * sendbuf, void * recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iallreduce_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Iallreduce_c(sendbuf, recvbuf, count, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iallreduce_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IALLREDUCE_C(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Iallreduce_c(sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype),
    MPI_Op_f2c(*op), MPI_Comm_f2c(*comm), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Iallreduce_c wrapper function (lowercase)
******************************************************/
void mpi_iallreduce_c(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLREDUCE_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallreduce_c wrapper function (lowercase_)
******************************************************/
void mpi_iallreduce_c_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLREDUCE_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iallreduce_c wrapper function (lowercase__)
******************************************************/
void mpi_iallreduce_c__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLREDUCE_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Allreduce_init wrapper function 
******************************************************/
int MPI_Allreduce_init(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allreduce_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allreduce_init(sendbuf, recvbuf, count, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allreduce_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLREDUCE_INIT(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Allreduce_init(sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), MPI_Comm_f2c(*comm),
    MPI_Info_f2c(*info), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Allreduce_init wrapper function (lowercase)
******************************************************/
void mpi_allreduce_init(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLREDUCE_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Allreduce_init wrapper function (lowercase_)
******************************************************/
void mpi_allreduce_init_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLREDUCE_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Allreduce_init wrapper function (lowercase__)
******************************************************/
void mpi_allreduce_init__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLREDUCE_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Allreduce_init_c wrapper function 
******************************************************/
int MPI_Allreduce_init_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Allreduce_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Allreduce_init_c(sendbuf, recvbuf, count, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Allreduce_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLREDUCE_INIT_C(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Allreduce_init_c(sendbuf, recvbuf, count, MPI_Type_f2c(*datatype),
    MPI_Op_f2c(*op), MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Allreduce_init_c wrapper function (lowercase)
******************************************************/
void mpi_allreduce_init_c(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLREDUCE_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Allreduce_init_c wrapper function (lowercase_)
******************************************************/
void mpi_allreduce_init_c_(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLREDUCE_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Allreduce_init_c wrapper function (lowercase__)
******************************************************/
void mpi_allreduce_init_c__(MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Count * count, MPI_Fint * datatype, MPI_Fint * op, MPI_Fint * comm,
MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLREDUCE_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Alltoall_c wrapper function 
******************************************************/
int MPI_Alltoall_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoall_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoall_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoall_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALL_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{

  *ierr = MPI_Alltoall_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, *recvcount,
    MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm));
  return ;
}

/******************************************************
***      MPI_Alltoall_c wrapper function (lowercase)
******************************************************/
void mpi_alltoall_c(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoall_c wrapper function (lowercase_)
******************************************************/
void mpi_alltoall_c_(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoall_c wrapper function (lowercase__)
******************************************************/
void mpi_alltoall_c__(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ialltoall_c wrapper function 
******************************************************/
int MPI_Ialltoall_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ialltoall_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ialltoall_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ialltoall_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IALLTOALL_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Ialltoall_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, *recvcount,
    MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Ialltoall_c wrapper function (lowercase)
******************************************************/
void mpi_ialltoall_c(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoall_c wrapper function (lowercase_)
******************************************************/
void mpi_ialltoall_c_(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoall_c wrapper function (lowercase__)
******************************************************/
void mpi_ialltoall_c__(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Alltoall_init wrapper function 
******************************************************/
int MPI_Alltoall_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoall_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoall_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoall_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALL_INIT(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Alltoall_init(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf, *recvcount,
    MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Alltoall_init wrapper function (lowercase)
******************************************************/
void mpi_alltoall_init(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALL_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Alltoall_init wrapper function (lowercase_)
******************************************************/
void mpi_alltoall_init_(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALL_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Alltoall_init wrapper function (lowercase__)
******************************************************/
void mpi_alltoall_init__(MPI_Aint * sendbuf, MPI_Fint * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Fint * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALL_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info, request,
    ierr);
  return ;
}


/******************************************************
***      MPI_Alltoall_init_c wrapper function 
******************************************************/
int MPI_Alltoall_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoall_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoall_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoall_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALL_INIT_C(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Alltoall_init_c(sendbuf, *sendcount, MPI_Type_f2c(*sendtype), recvbuf,
    *recvcount, MPI_Type_f2c(*recvtype),  MPI_Info_f2c(*info), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Alltoall_init_c wrapper function (lowercase)
******************************************************/
void mpi_alltoall_init_c(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALL_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoall_init_c wrapper function (lowercase_)
******************************************************/
void mpi_alltoall_init_c_(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALL_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoall_init_c wrapper function (lowercase__)
******************************************************/
void mpi_alltoall_init_c__(MPI_Aint * sendbuf, MPI_Count * sendcount, MPI_Fint * sendtype, MPI_Aint * recvbuf, MPI_Count * recvcount,
MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALL_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Alltoallv_c wrapper function 
******************************************************/
int MPI_Alltoallv_c(TAU_MPICH3_CONST void * sendbuf, TAU_MPICH3_CONST MPI_Count * sendcounts, TAU_MPICH3_CONST MPI_Aint * sdispls, 
MPI_Datatype sendtype, void * recvbuf, TAU_MPICH3_CONST MPI_Count * recvcounts, TAU_MPICH3_CONST MPI_Aint * rdispls, 
MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoallv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoallv_c(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoallv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALLV_C(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{

  *ierr = MPI_Alltoallv_c(sendbuf, sendcounts, sdispls, MPI_Type_f2c(*sendtype), recvbuf,
    recvcounts, rdispls, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm));
  return ;
}

/******************************************************
***      MPI_Alltoallv_c wrapper function (lowercase)
******************************************************/
void mpi_alltoallv_c(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallv_c wrapper function (lowercase_)
******************************************************/
void mpi_alltoallv_c_(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallv_c wrapper function (lowercase__)
******************************************************/
void mpi_alltoallv_c__(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ialltoallv_c wrapper function 
******************************************************/
int MPI_Ialltoallv_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count * sendcounts, TAU_MPICH3_CONST MPI_Aint * sdispls, 
MPI_Datatype sendtype,void* recvbuf, TAU_MPICH3_CONST MPI_Count * recvcounts, TAU_MPICH3_CONST MPI_Aint * rdispls, 
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ialltoallv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ialltoallv_c(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ialltoallv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IALLTOALLV_C(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, * MPI_Aint * rdispls, MPI_Fint * recvtype,
MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Ialltoallv_c(sendbuf, sendcounts, sdispls, MPI_Type_f2c(*sendtype), recvbuf,
    recvcounts, rdispls, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Ialltoallv_c wrapper function (lowercase)
******************************************************/
void mpi_ialltoallv_c(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, * MPI_Aint * rdispls, MPI_Fint * recvtype,
MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoallv_c wrapper function (lowercase_)
******************************************************/
void mpi_ialltoallv_c_(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, * MPI_Aint * rdispls, MPI_Fint * recvtype,
MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoallv_c wrapper function (lowercase__)
******************************************************/
void mpi_ialltoallv_c__(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, * MPI_Aint * rdispls, MPI_Fint * recvtype,
MPI_Fint * comm, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_IALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Alltoallv_init wrapper function 
******************************************************/
int MPI_Alltoallv_init(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int * sendcounts, TAU_MPICH3_CONST * int sdispls, MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST int * recvcounts, TAU_MPICH3_CONST int * rdispls, MPI_Datatype recvtype, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoallv_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoallv_init(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoallv_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALLV_INIT(MPI_Aint * sendbuf, int * sendcounts, int * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
int * recvcounts, int * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Alltoallv_init(sendbuf, sendcounts, sdispls, MPI_Type_f2c(*sendtype), recvbuf,
    recvcounts, rdispls, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Alltoallv_init wrapper function (lowercase)
******************************************************/
void mpi_alltoallv_init(MPI_Aint * sendbuf, int * sendcounts, int * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
int * recvcounts, int * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_INIT(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallv_init wrapper function (lowercase_)
******************************************************/
void mpi_alltoallv_init_(MPI_Aint * sendbuf, int * sendcounts, int * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
int * recvcounts, int * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_INIT(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallv_init wrapper function (lowercase__)
******************************************************/
void mpi_alltoallv_init__(MPI_Aint * sendbuf, int * sendcounts, int * sdispls, MPI_Fint * sendtype, MPI_Aint * recvbuf,
int * recvcounts, int * rdispls, MPI_Fint * recvtype, MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_INIT(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
    comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Alltoallv_init_c wrapper function 
******************************************************/
int MPI_Alltoallv_init_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count * sendcounts, TAU_MPICH3_CONST MPI_Aint * sdispls, MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST MPI_Count * recvcounts, TAU_MPICH3_CONST MPI_Aint * rdispls, MPI_Datatype recvtype,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoallv_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoallv_init_c(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoallv_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALLV_INIT_C(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtype,
MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Alltoallv_init_c(sendbuf, sendcounts, sdispls, MPI_Type_f2c(*sendtype),
    recvbuf, recvcounts, rdispls, MPI_Type_f2c(*recvtype), MPI_Comm_f2c(*comm), MPI_Info_f2c(*info), &local_request);
  *request = MPI_Request_c2f(local_request);
  return ;
}

/******************************************************
***      MPI_Alltoallv_init_c wrapper function (lowercase)
******************************************************/
void mpi_alltoallv_init_c(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtype,
MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_INIT_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallv_init_c wrapper function (lowercase_)
******************************************************/
void mpi_alltoallv_init_c_(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtype,
MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_INIT_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallv_init_c wrapper function (lowercase__)
******************************************************/
void mpi_alltoallv_init_c__(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Fint * sendtype,
MPI_Aint * recvbuf, MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Fint * recvtype,
MPI_Fint * comm, MPI_Fint * info, MPI_Request * request, MPI_Fint * ierr)
{
  MPI_ALLTOALLV_INIT_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Alltoallw_c wrapper function 
******************************************************/
int MPI_Alltoallw_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count * sendcounts, TAU_MPICH3_CONST MPI_Aint * sdispls,
TAU_MPICH3_CONST MPI_Datatype * sendtypes, void* recvbuf, TAU_MPICH3_CONST MPI_Count * recvcounts,
TAU_MPICH3_CONST MPI_Aint * rdispls, TAU_MPICH3_CONST MPI_Datatype * recvtypes, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoallw_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoallw_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoallw_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALLW_C(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Datatype * sendtypes, MPI_Aint * recvbuf, 
MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Datatype * recvtypes, MPI_Fint * comm, MPI_Fint * ierr)
{
  int comm_size;
  MPI_Comm local_comm = MPI_Comm_f2c(*comm);
  MPI_Comm_size(local_comm, &comm_size);
  TAU_DECL_LOCAL(MPI_Datatype, local_send_types);
  TAU_DECL_ALLOC_LOCAL(MPI_Datatype, local_recv_types, comm_size);
  TAU_ALLOC_LOCAL(MPI_Datatype, local_send_types, comm_size);
  TAU_ASSIGN_VALUES(local_send_types, sendtypes, comm_size, MPI_Type_f2c);
  TAU_ASSIGN_VALUES(local_recv_types, recvtypes, comm_size, MPI_Type_f2c);
  *ierr = MPI_Alltoallw_c(sendbuf, sendcounts, sdispls, local_send_types, recvbuf, recvcounts, rdispls,
    local_recv_types,  local_comm);
  TAU_FREE_LOCAL(local_send_types);
  TAU_FREE_LOCAL(local_recv_types); 
  return ;
}

/******************************************************
***      MPI_Alltoallw_c wrapper function (lowercase)
******************************************************/
void mpi_alltoallw_c(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Datatype * sendtypes, MPI_Aint * recvbuf, 
MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Datatype * recvtypes, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallw_c wrapper function (lowercase_)
******************************************************/
void mpi_alltoallw_c_(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Datatype * sendtypes, MPI_Aint * recvbuf, 
MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Datatype * recvtypes, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallw_c wrapper function (lowercase__)
******************************************************/
void mpi_alltoallw_c__(MPI_Aint * sendbuf, MPI_Count * sendcounts, MPI_Aint * sdispls, MPI_Datatype * sendtypes, MPI_Aint * recvbuf, 
MPI_Count * recvcounts, MPI_Aint * rdispls, MPI_Datatype * recvtypes, MPI_Fint * comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ialltoallw_c wrapper function 
******************************************************/
int MPI_Ialltoallw_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[],
TAU_MPICH3_CONST MPI_Datatype sendtypes[], void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[],
TAU_MPICH3_CONST MPI_Aint rdispls[], TAU_MPICH3_CONST MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ialltoallw_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ialltoallw_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ialltoallw_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IALLTOALLW_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Request* request,
MPI_Fint * ierr)
{
  int comm_size;
  MPI_Comm local_comm = MPI_Comm_f2c(*comm);
  MPI_Comm_size(local_comm, &comm_size);
  TAU_DECL_LOCAL(MPI_Datatype, local_send_types);
  TAU_DECL_ALLOC_LOCAL(MPI_Datatype, local_recv_types, comm_size);
  TAU_ALLOC_LOCAL(MPI_Datatype, local_send_types, comm_size);
  TAU_ASSIGN_VALUES(local_send_types, sendtypes, comm_size, MPI_Type_f2c);
  TAU_ASSIGN_VALUES(local_recv_types, recvtypes, comm_size, MPI_Type_f2c);
  MPI_Request local_request;
  *ierr = MPI_Ialltoallw_c(sendbuf, sendcounts, sdispls, local_send_types, recvbuf, recvcounts, rdispls,
    local_send_types, local_comm, &local_request);
  *request = MPI_Request_c2f(local_request);
  TAU_FREE_LOCAL(local_send_types);
  TAU_FREE_LOCAL(local_recv_types); 
  return ;
}

/******************************************************
***      MPI_Ialltoallw_c wrapper function (lowercase)
******************************************************/
void mpi_ialltoallw_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_IALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoallw_c wrapper function (lowercase_)
******************************************************/
void mpi_ialltoallw_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_IALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ialltoallw_c wrapper function (lowercase__)
******************************************************/
void mpi_ialltoallw_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_IALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes,
    comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Alltoallw_init wrapper function 
******************************************************/
int MPI_Alltoallw_init(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int sendcounts[], TAU_MPICH3_CONST int sdispls[], TAU_MPICH3_CONST MPI_Datatype sendtypes[],
void* recvbuf, TAU_MPICH3_CONST int recvcounts[], TAU_MPICH3_CONST int rdispls[], TAU_MPICH3_CONST MPI_Datatype recvtypes[],
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoallw_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoallw_init(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoallw_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALLW_INIT(MPI_Aint sendbuf, int sendcounts[], int sdispls[], MPI_Datatype sendtypes[],
MPI_Aint recvbuf, int recvcounts[], int rdispls[], MPI_Datatype recvtypes[],
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Alltoallw_init(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Alltoallw_init wrapper function (lowercase)
******************************************************/
void mpi_alltoallw_init(MPI_Aint sendbuf, int sendcounts[], int sdispls[], MPI_Datatype sendtypes[],
MPI_Aint recvbuf, int recvcounts[], int rdispls[], MPI_Datatype recvtypes[],
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_INIT(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallw_init wrapper function (lowercase_)
******************************************************/
void mpi_alltoallw_init_(MPI_Aint sendbuf, int sendcounts[], int sdispls[], MPI_Datatype sendtypes[],
MPI_Aint recvbuf, int recvcounts[], int rdispls[], MPI_Datatype recvtypes[],
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_INIT(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallw_init wrapper function (lowercase__)
******************************************************/
void mpi_alltoallw_init__(MPI_Aint sendbuf, int sendcounts[], int sdispls[], MPI_Datatype sendtypes[],
MPI_Aint recvbuf, int recvcounts[], int rdispls[], MPI_Datatype recvtypes[],
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_INIT(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Alltoallw_init_c wrapper function 
******************************************************/
int MPI_Alltoallw_init_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[],
TAU_MPICH3_CONST MPI_Datatype sendtypes[], void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[],
TAU_MPICH3_CONST MPI_Aint rdispls[], TAU_MPICH3_CONST MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Info info,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Alltoallw_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Alltoallw_init_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Alltoallw_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ALLTOALLW_INIT_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Alltoallw_init_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Alltoallw_init_c wrapper function (lowercase)
******************************************************/
void mpi_alltoallw_init_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_INIT_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallw_init_c wrapper function (lowercase_)
******************************************************/
void mpi_alltoallw_init_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_INIT_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Alltoallw_init_c wrapper function (lowercase__)
******************************************************/
void mpi_alltoallw_init_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ALLTOALLW_INIT_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Barrier_init wrapper function 
******************************************************/
int MPI_Barrier_init(MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Barrier_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Barrier_init(comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Barrier_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_BARRIER_INIT(MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Barrier_init(/* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Barrier_init wrapper function (lowercase)
******************************************************/
void mpi_barrier_init(MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BARRIER_INIT(comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Barrier_init wrapper function (lowercase_)
******************************************************/
void mpi_barrier_init_(MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BARRIER_INIT(comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Barrier_init wrapper function (lowercase__)
******************************************************/
void mpi_barrier_init__(MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BARRIER_INIT(comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Bcast_c wrapper function 
******************************************************/
int MPI_Bcast_c(void* buffer, MPI_Count count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Bcast_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Bcast_c(buffer, count, datatype, root, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Bcast_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_BCAST_C(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Bcast_c(buffer, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Bcast_c wrapper function (lowercase)
******************************************************/
void mpi_bcast_c(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_BCAST_C(buffer, count, datatype, root, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Bcast_c wrapper function (lowercase_)
******************************************************/
void mpi_bcast_c_(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_BCAST_C(buffer, count, datatype, root, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Bcast_c wrapper function (lowercase__)
******************************************************/
void mpi_bcast_c__(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_BCAST_C(buffer, count, datatype, root, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ibcast_c wrapper function 
******************************************************/
int MPI_Ibcast_c(void* buffer, MPI_Count count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ibcast_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ibcast_c(buffer, count, datatype, root, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ibcast_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IBCAST_C(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ibcast_c(buffer, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ibcast_c wrapper function (lowercase)
******************************************************/
void mpi_ibcast_c(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IBCAST_C(buffer, count, datatype, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ibcast_c wrapper function (lowercase_)
******************************************************/
void mpi_ibcast_c_(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IBCAST_C(buffer, count, datatype, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ibcast_c wrapper function (lowercase__)
******************************************************/
void mpi_ibcast_c__(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IBCAST_C(buffer, count, datatype, root, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Bcast_init wrapper function 
******************************************************/
int MPI_Bcast_init(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Info info,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Bcast_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Bcast_init(buffer, count, datatype, root, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Bcast_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_BCAST_INIT(MPI_Aint buffer, MPI_Fint count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Bcast_init(buffer, /* MPI_HANDLE_TYPES */ count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Bcast_init wrapper function (lowercase)
******************************************************/
void mpi_bcast_init(MPI_Aint buffer, MPI_Fint count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BCAST_INIT(buffer, count, datatype, root, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Bcast_init wrapper function (lowercase_)
******************************************************/
void mpi_bcast_init_(MPI_Aint buffer, MPI_Fint count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BCAST_INIT(buffer, count, datatype, root, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Bcast_init wrapper function (lowercase__)
******************************************************/
void mpi_bcast_init__(MPI_Aint buffer, MPI_Fint count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BCAST_INIT(buffer, count, datatype, root, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Bcast_init_c wrapper function 
******************************************************/
int MPI_Bcast_init_c(void* buffer, MPI_Count count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Info info,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Bcast_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Bcast_init_c(buffer, count, datatype, root, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Bcast_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_BCAST_INIT_C(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Bcast_init_c(buffer, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Bcast_init_c wrapper function (lowercase)
******************************************************/
void mpi_bcast_init_c(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BCAST_INIT_C(buffer, count, datatype, root, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Bcast_init_c wrapper function (lowercase_)
******************************************************/
void mpi_bcast_init_c_(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BCAST_INIT_C(buffer, count, datatype, root, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Bcast_init_c wrapper function (lowercase__)
******************************************************/
void mpi_bcast_init_c__(MPI_Aint buffer, MPI_Count count, MPI_Fint datatype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BCAST_INIT_C(buffer, count, datatype, root, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Bsend_c wrapper function 
******************************************************/
int MPI_Bsend_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Bsend_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Bsend_c(buf, count, datatype, dest, tag, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Bsend_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_BSEND_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Bsend_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Bsend_c wrapper function (lowercase)
******************************************************/
void mpi_bsend_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_BSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Bsend_c wrapper function (lowercase_)
******************************************************/
void mpi_bsend_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_BSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Bsend_c wrapper function (lowercase__)
******************************************************/
void mpi_bsend_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_BSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Bsend_init_c wrapper function 
******************************************************/
int MPI_Bsend_init_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Bsend_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Bsend_init_c(buf, count, datatype, dest, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Bsend_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_BSEND_INIT_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Bsend_init_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Bsend_init_c wrapper function (lowercase)
******************************************************/
void mpi_bsend_init_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Bsend_init_c wrapper function (lowercase_)
******************************************************/
void mpi_bsend_init_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Bsend_init_c wrapper function (lowercase__)
******************************************************/
void mpi_bsend_init_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_BSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Buffer_attach_c wrapper function 
******************************************************/
int MPI_Buffer_attach_c(void* buffer, MPI_Count size)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Buffer_attach_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Buffer_attach_c(buffer, size);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Buffer_attach_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_BUFFER_ATTACH_C(MPI_Aint buffer, MPI_Count size, MPI_Fint * ierr)
{

  *ierr = MPI_Buffer_attach_c(buffer, size);
  return ;
}

/******************************************************
***      MPI_Buffer_attach_c wrapper function (lowercase)
******************************************************/
void mpi_buffer_attach_c(MPI_Aint buffer, MPI_Count size, MPI_Fint * ierr)
{
  MPI_BUFFER_ATTACH_C(buffer, size, ierr);
  return ;
}

/******************************************************
***      MPI_Buffer_attach_c wrapper function (lowercase_)
******************************************************/
void mpi_buffer_attach_c_(MPI_Aint buffer, MPI_Count size, MPI_Fint * ierr)
{
  MPI_BUFFER_ATTACH_C(buffer, size, ierr);
  return ;
}

/******************************************************
***      MPI_Buffer_attach_c wrapper function (lowercase__)
******************************************************/
void mpi_buffer_attach_c__(MPI_Aint buffer, MPI_Count size, MPI_Fint * ierr)
{
  MPI_BUFFER_ATTACH_C(buffer, size, ierr);
  return ;
}


/******************************************************
***      MPI_Buffer_detach_c wrapper function 
******************************************************/
int MPI_Buffer_detach_c(void* buffer_addr, MPI_Count* size)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Buffer_detach_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Buffer_detach_c(buffer_addr, size);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Buffer_detach_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_BUFFER_DETACH_C(MPI_Aint buffer_addr, MPI_Count* size, MPI_Fint * ierr)
{

  *ierr = MPI_Buffer_detach_c(buffer_addr, size);
  return ;
}

/******************************************************
***      MPI_Buffer_detach_c wrapper function (lowercase)
******************************************************/
void mpi_buffer_detach_c(MPI_Aint buffer_addr, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_BUFFER_DETACH_C(buffer_addr, size, ierr);
  return ;
}

/******************************************************
***      MPI_Buffer_detach_c wrapper function (lowercase_)
******************************************************/
void mpi_buffer_detach_c_(MPI_Aint buffer_addr, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_BUFFER_DETACH_C(buffer_addr, size, ierr);
  return ;
}

/******************************************************
***      MPI_Buffer_detach_c wrapper function (lowercase__)
******************************************************/
void mpi_buffer_detach_c__(MPI_Aint buffer_addr, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_BUFFER_DETACH_C(buffer_addr, size, ierr);
  return ;
}


/******************************************************
***      MPI_Exscan_c wrapper function 
******************************************************/
int MPI_Exscan_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Exscan_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Exscan_c(sendbuf, recvbuf, count, datatype, op, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Exscan_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_EXSCAN_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Exscan_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Exscan_c wrapper function (lowercase)
******************************************************/
void mpi_exscan_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_EXSCAN_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Exscan_c wrapper function (lowercase_)
******************************************************/
void mpi_exscan_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_EXSCAN_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Exscan_c wrapper function (lowercase__)
******************************************************/
void mpi_exscan_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_EXSCAN_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Iexscan_c wrapper function 
******************************************************/
int MPI_Iexscan_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iexscan_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Iexscan_c(sendbuf, recvbuf, count, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iexscan_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IEXSCAN_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Iexscan_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Iexscan_c wrapper function (lowercase)
******************************************************/
void mpi_iexscan_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IEXSCAN_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iexscan_c wrapper function (lowercase_)
******************************************************/
void mpi_iexscan_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IEXSCAN_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iexscan_c wrapper function (lowercase__)
******************************************************/
void mpi_iexscan_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IEXSCAN_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Exscan_init wrapper function 
******************************************************/
int MPI_Exscan_init(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Exscan_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Exscan_init(sendbuf, recvbuf, count, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Exscan_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_EXSCAN_INIT(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Exscan_init(sendbuf, recvbuf, /* MPI_HANDLE_TYPES */ count,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Exscan_init wrapper function (lowercase)
******************************************************/
void mpi_exscan_init(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_EXSCAN_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Exscan_init wrapper function (lowercase_)
******************************************************/
void mpi_exscan_init_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_EXSCAN_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Exscan_init wrapper function (lowercase__)
******************************************************/
void mpi_exscan_init__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_EXSCAN_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Exscan_init_c wrapper function 
******************************************************/
int MPI_Exscan_init_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Exscan_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Exscan_init_c(sendbuf, recvbuf, count, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Exscan_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_EXSCAN_INIT_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Exscan_init_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Exscan_init_c wrapper function (lowercase)
******************************************************/
void mpi_exscan_init_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_EXSCAN_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Exscan_init_c wrapper function (lowercase_)
******************************************************/
void mpi_exscan_init_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_EXSCAN_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Exscan_init_c wrapper function (lowercase__)
******************************************************/
void mpi_exscan_init_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_EXSCAN_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iread_c wrapper function 
******************************************************/
int MPI_File_iread_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iread_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iread_c(fh, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iread_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IREAD_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iread_c(/* MPI_HANDLE_TYPES */ fh, buf, count, /* MPI_HANDLE_TYPES */ datatype,
    request);
  return ;
}

/******************************************************
***      MPI_File_iread_c wrapper function (lowercase)
******************************************************/
void mpi_file_iread_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iread_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iread_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_C(fh, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iread_all_c wrapper function 
******************************************************/
int MPI_File_iread_all_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iread_all_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iread_all_c(fh, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iread_all_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IREAD_ALL_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iread_all_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, request);
  return ;
}

/******************************************************
***      MPI_File_iread_all_c wrapper function (lowercase)
******************************************************/
void mpi_file_iread_all_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_ALL_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_all_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iread_all_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_ALL_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_all_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iread_all_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_ALL_C(fh, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iread_at_c wrapper function 
******************************************************/
int MPI_File_iread_at_c(MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count, MPI_Datatype datatype,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iread_at_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iread_at_c(fh, offset, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iread_at_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IREAD_AT_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iread_at_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, request);
  return ;
}

/******************************************************
***      MPI_File_iread_at_c wrapper function (lowercase)
******************************************************/
void mpi_file_iread_at_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_at_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iread_at_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_at_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iread_at_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_Comm_create_from_group wrapper function 
******************************************************/
int MPI_Comm_create_from_group(MPI_Group group, TAU_MPICH3_CONST char* stringtag, MPI_Info info, MPI_Errhandler errhandler, MPI_Comm* newcomm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Comm_create_from_group()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Comm_create_from_group(group, stringtag, info, errhandler, newcomm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Comm_create_from_group wrapper function (uppercase Fortran)
******************************************************/
void MPI_COMM_CREATE_FROM_GROUP(MPI_Fint group, char* stringtag, MPI_Fint info, MPI_Fint errhandler, MPI_Comm* newcomm,
MPI_Fint * ierr)
{

  *ierr = MPI_Comm_create_from_group(/* MPI_HANDLE_TYPES */ group, stringtag,
    /* MPI_HANDLE_TYPES */ info, /* MPI_HANDLE_TYPES */ errhandler, newcomm);
  return ;
}

/******************************************************
***      MPI_Comm_create_from_group wrapper function (lowercase)
******************************************************/
void mpi_comm_create_from_group(MPI_Fint group, char* stringtag, MPI_Fint info, MPI_Fint errhandler, MPI_Comm* newcomm,
MPI_Fint * ierr)
{
  MPI_COMM_CREATE_FROM_GROUP(group, stringtag, info, errhandler, newcomm, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_create_from_group wrapper function (lowercase_)
******************************************************/
void mpi_comm_create_from_group_(MPI_Fint group, char* stringtag, MPI_Fint info, MPI_Fint errhandler, MPI_Comm* newcomm,
MPI_Fint * ierr)
{
  MPI_COMM_CREATE_FROM_GROUP(group, stringtag, info, errhandler, newcomm, ierr);
  return ;
}

/******************************************************
***      MPI_Comm_create_from_group wrapper function (lowercase__)
******************************************************/
void mpi_comm_create_from_group__(MPI_Fint group, char* stringtag, MPI_Fint info, MPI_Fint errhandler, MPI_Comm* newcomm,
MPI_Fint * ierr)
{
  MPI_COMM_CREATE_FROM_GROUP(group, stringtag, info, errhandler, newcomm, ierr);
  return ;
}


/******************************************************
***      MPI_Group_from_session_pset wrapper function 
******************************************************/
int MPI_Group_from_session_pset(MPI_Session session, TAU_MPICH3_CONST char* pset_name, MPI_Group* newgroup)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Group_from_session_pset()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Group_from_session_pset(session, pset_name, newgroup);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Group_from_session_pset wrapper function (uppercase Fortran)
******************************************************/
void MPI_GROUP_FROM_SESSION_PSET(MPI_Fint session, char* pset_name, MPI_Group* newgroup, MPI_Fint * ierr)
{

  *ierr = MPI_Group_from_session_pset(/* MPI_HANDLE_TYPES */ session, pset_name, newgroup);
  return ;
}

/******************************************************
***      MPI_Group_from_session_pset wrapper function (lowercase)
******************************************************/
void mpi_group_from_session_pset(MPI_Fint session, char* pset_name, MPI_Group* newgroup, MPI_Fint * ierr)
{
  MPI_GROUP_FROM_SESSION_PSET(session, pset_name, newgroup, ierr);
  return ;
}

/******************************************************
***      MPI_Group_from_session_pset wrapper function (lowercase_)
******************************************************/
void mpi_group_from_session_pset_(MPI_Fint session, char* pset_name, MPI_Group* newgroup, MPI_Fint * ierr)
{
  MPI_GROUP_FROM_SESSION_PSET(session, pset_name, newgroup, ierr);
  return ;
}

/******************************************************
***      MPI_Group_from_session_pset wrapper function (lowercase__)
******************************************************/
void mpi_group_from_session_pset__(MPI_Fint session, char* pset_name, MPI_Group* newgroup, MPI_Fint * ierr)
{
  MPI_GROUP_FROM_SESSION_PSET(session, pset_name, newgroup, ierr);
  return ;
}


/******************************************************
***      MPI_Intercomm_create_from_groups wrapper function 
******************************************************/
int MPI_Intercomm_create_from_groups(MPI_Group local_group, int local_leader, MPI_Group remote_group, int remote_leader,
TAU_MPICH3_CONST char* stringtag, MPI_Info info, MPI_Errhandler errhandler, MPI_Comm* newintercomm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Intercomm_create_from_groups()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Intercomm_create_from_groups(local_group, local_leader, remote_group, remote_leader,
    stringtag, info, errhandler, newintercomm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Intercomm_create_from_groups wrapper function (uppercase Fortran)
******************************************************/
void MPI_INTERCOMM_CREATE_FROM_GROUPS(MPI_Fint local_group, MPI_Fint local_leader, MPI_Fint remote_group, MPI_Fint remote_leader,
char* stringtag, MPI_Fint info, MPI_Fint errhandler, MPI_Comm* newintercomm, MPI_Fint * ierr)
{

  *ierr = MPI_Intercomm_create_from_groups(/* MPI_HANDLE_TYPES */ local_group,
    /* MPI_HANDLE_TYPES */ local_leader, /* MPI_HANDLE_TYPES */ remote_group,
    /* MPI_HANDLE_TYPES */ remote_leader, stringtag, /* MPI_HANDLE_TYPES */ info,
    /* MPI_HANDLE_TYPES */ errhandler, newintercomm);
  return ;
}

/******************************************************
***      MPI_Intercomm_create_from_groups wrapper function (lowercase)
******************************************************/
void mpi_intercomm_create_from_groups(MPI_Fint local_group, MPI_Fint local_leader, MPI_Fint remote_group, MPI_Fint remote_leader,
char* stringtag, MPI_Fint info, MPI_Fint errhandler, MPI_Comm* newintercomm, MPI_Fint * ierr)
{
  MPI_INTERCOMM_CREATE_FROM_GROUPS(local_group, local_leader, remote_group, remote_leader,
    stringtag, info, errhandler, newintercomm, ierr);
  return ;
}

/******************************************************
***      MPI_Intercomm_create_from_groups wrapper function (lowercase_)
******************************************************/
void mpi_intercomm_create_from_groups_(MPI_Fint local_group, MPI_Fint local_leader, MPI_Fint remote_group, MPI_Fint remote_leader,
char* stringtag, MPI_Fint info, MPI_Fint errhandler, MPI_Comm* newintercomm, MPI_Fint * ierr)
{
  MPI_INTERCOMM_CREATE_FROM_GROUPS(local_group, local_leader, remote_group, remote_leader,
    stringtag, info, errhandler, newintercomm, ierr);
  return ;
}

/******************************************************
***      MPI_Intercomm_create_from_groups wrapper function (lowercase__)
******************************************************/
void mpi_intercomm_create_from_groups__(MPI_Fint local_group, MPI_Fint local_leader, MPI_Fint remote_group, MPI_Fint remote_leader,
char* stringtag, MPI_Fint info, MPI_Fint errhandler, MPI_Comm* newintercomm, MPI_Fint * ierr)
{
  MPI_INTERCOMM_CREATE_FROM_GROUPS(local_group, local_leader, remote_group, remote_leader,
    stringtag, info, errhandler, newintercomm, ierr);
  return ;
}


/******************************************************
***      MPI_Isendrecv_c wrapper function 
******************************************************/
int MPI_Isendrecv_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, int dest, int sendtag,
void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Isendrecv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Isendrecv_c(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
    source, recvtag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Isendrecv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISENDRECV_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Fint dest, MPI_Fint sendtag,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint source, MPI_Fint recvtag,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Isendrecv_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype,
    /* MPI_HANDLE_TYPES */ dest, /* MPI_HANDLE_TYPES */ sendtag, recvbuf, recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ source, /* MPI_HANDLE_TYPES */ recvtag,
    /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Isendrecv_c wrapper function (lowercase)
******************************************************/
void mpi_isendrecv_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Fint dest, MPI_Fint sendtag,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint source, MPI_Fint recvtag,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_C(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isendrecv_c wrapper function (lowercase_)
******************************************************/
void mpi_isendrecv_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Fint dest, MPI_Fint sendtag,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint source, MPI_Fint recvtag,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_C(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isendrecv_c wrapper function (lowercase__)
******************************************************/
void mpi_isendrecv_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Fint dest, MPI_Fint sendtag,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint source, MPI_Fint recvtag,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_C(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Isendrecv_replace_c wrapper function 
******************************************************/
int MPI_Isendrecv_replace_c(void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Isendrecv_replace_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Isendrecv_replace_c(buf, count, datatype, dest, sendtag, source, recvtag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Isendrecv_replace_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISENDRECV_REPLACE_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint sendtag, MPI_Fint source,
MPI_Fint recvtag, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Isendrecv_replace_c(buf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ dest, /* MPI_HANDLE_TYPES */ sendtag, /* MPI_HANDLE_TYPES */ source,
    /* MPI_HANDLE_TYPES */ recvtag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Isendrecv_replace_c wrapper function (lowercase)
******************************************************/
void mpi_isendrecv_replace_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint sendtag, MPI_Fint source,
MPI_Fint recvtag, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_REPLACE_C(buf, count, datatype, dest, sendtag, source, recvtag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isendrecv_replace_c wrapper function (lowercase_)
******************************************************/
void mpi_isendrecv_replace_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint sendtag, MPI_Fint source,
MPI_Fint recvtag, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_REPLACE_C(buf, count, datatype, dest, sendtag, source, recvtag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isendrecv_replace_c wrapper function (lowercase__)
******************************************************/
void mpi_isendrecv_replace_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint sendtag, MPI_Fint source,
MPI_Fint recvtag, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISENDRECV_REPLACE_C(buf, count, datatype, dest, sendtag, source, recvtag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iread_at_all_c wrapper function 
******************************************************/
int MPI_File_iread_at_all_c(MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count, MPI_Datatype datatype,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iread_at_all_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iread_at_all_c(fh, offset, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iread_at_all_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IREAD_AT_ALL_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iread_at_all_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, request);
  return ;
}

/******************************************************
***      MPI_File_iread_at_all_c wrapper function (lowercase)
******************************************************/
void mpi_file_iread_at_all_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_ALL_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_at_all_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iread_at_all_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_ALL_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_at_all_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iread_at_all_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT_ALL_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iread_shared_c wrapper function 
******************************************************/
int MPI_File_iread_shared_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iread_shared_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iread_shared_c(fh, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iread_shared_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IREAD_SHARED_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iread_shared_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, request);
  return ;
}

/******************************************************
***      MPI_File_iread_shared_c wrapper function (lowercase)
******************************************************/
void mpi_file_iread_shared_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_shared_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iread_shared_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iread_shared_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iread_shared_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED_C(fh, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iwrite_c wrapper function 
******************************************************/
int MPI_File_iwrite_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iwrite_c(fh, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iwrite_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IWRITE_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iwrite_c(/* MPI_HANDLE_TYPES */ fh, buf, count, /* MPI_HANDLE_TYPES */ datatype,
    request);
  return ;
}

/******************************************************
***      MPI_File_iwrite_c wrapper function (lowercase)
******************************************************/
void mpi_file_iwrite_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iwrite_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iwrite_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_C(fh, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iwrite_all_c wrapper function 
******************************************************/
int MPI_File_iwrite_all_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_all_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iwrite_all_c(fh, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iwrite_all_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IWRITE_ALL_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iwrite_all_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, request);
  return ;
}

/******************************************************
***      MPI_File_iwrite_all_c wrapper function (lowercase)
******************************************************/
void mpi_file_iwrite_all_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_ALL_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_all_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iwrite_all_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_ALL_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_all_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iwrite_all_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_ALL_C(fh, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iwrite_at_c wrapper function 
******************************************************/
int MPI_File_iwrite_at_c(MPI_File fh, MPI_Offset offset, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_at_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iwrite_at_c(fh, offset, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iwrite_at_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IWRITE_AT_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iwrite_at_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, request);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_c wrapper function (lowercase)
******************************************************/
void mpi_file_iwrite_at_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iwrite_at_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iwrite_at_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iwrite_at_all_c wrapper function 
******************************************************/
int MPI_File_iwrite_at_all_c(MPI_File fh, MPI_Offset offset, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_at_all_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iwrite_at_all_c(fh, offset, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iwrite_at_all_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IWRITE_AT_ALL_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iwrite_at_all_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, request);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_all_c wrapper function (lowercase)
******************************************************/
void mpi_file_iwrite_at_all_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_ALL_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_all_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iwrite_at_all_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_ALL_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_at_all_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iwrite_at_all_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT_ALL_C(fh, offset, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_iwrite_shared_c wrapper function 
******************************************************/
int MPI_File_iwrite_shared_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_shared_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_iwrite_shared_c(fh, buf, count, datatype, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_iwrite_shared_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_IWRITE_SHARED_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_File_iwrite_shared_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, request);
  return ;
}

/******************************************************
***      MPI_File_iwrite_shared_c wrapper function (lowercase)
******************************************************/
void mpi_file_iwrite_shared_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_shared_c wrapper function (lowercase_)
******************************************************/
void mpi_file_iwrite_shared_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED_C(fh, buf, count, datatype, request, ierr);
  return ;
}

/******************************************************
***      MPI_File_iwrite_shared_c wrapper function (lowercase__)
******************************************************/
void mpi_file_iwrite_shared_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED_C(fh, buf, count, datatype, request, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_c wrapper function 
******************************************************/
int MPI_File_read_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_c(fh, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_File_read_c(/* MPI_HANDLE_TYPES */ fh, buf, count, /* MPI_HANDLE_TYPES */ datatype,
    status);
  return ;
}

/******************************************************
***      MPI_File_read_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_C(fh, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_all_c wrapper function 
******************************************************/
int MPI_File_read_all_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_all_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_all_c(fh, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_all_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_ALL_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_File_read_all_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_read_all_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_all_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_all_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_all_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_all_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_all_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_C(fh, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_all_begin_c wrapper function 
******************************************************/
int MPI_File_read_all_begin_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_all_begin_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_all_begin_c(fh, buf, count, datatype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_all_begin_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_ALL_BEGIN_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{

  *ierr = MPI_File_read_all_begin_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype);
  return ;
}

/******************************************************
***      MPI_File_read_all_begin_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_all_begin_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_all_begin_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_all_begin_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_all_begin_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_all_begin_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_at_c wrapper function 
******************************************************/
int MPI_File_read_at_c(MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count, MPI_Datatype datatype,
MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_at_c(fh, offset, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_at_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_AT_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{

  *ierr = MPI_File_read_at_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_read_at_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_at_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_at_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_at_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_at_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_at_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_at_all_c wrapper function 
******************************************************/
int MPI_File_read_at_all_c(MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count, MPI_Datatype datatype,
MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_all_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_at_all_c(fh, offset, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_at_all_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_AT_ALL_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{

  *ierr = MPI_File_read_at_all_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_read_at_all_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_at_all_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_at_all_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_at_all_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_at_all_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_at_all_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_at_all_begin_c wrapper function 
******************************************************/
int MPI_File_read_at_all_begin_c(MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count, MPI_Datatype datatype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_all_begin_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_at_all_begin_c(fh, offset, buf, count, datatype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_at_all_begin_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_AT_ALL_BEGIN_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{

  *ierr = MPI_File_read_at_all_begin_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype);
  return ;
}

/******************************************************
***      MPI_File_read_at_all_begin_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_at_all_begin_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_BEGIN_C(fh, offset, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_at_all_begin_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_at_all_begin_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_BEGIN_C(fh, offset, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_at_all_begin_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_at_all_begin_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_BEGIN_C(fh, offset, buf, count, datatype, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_ordered_c wrapper function 
******************************************************/
int MPI_File_read_ordered_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_ordered_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_ordered_c(fh, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_ordered_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_ORDERED_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_File_read_ordered_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_read_ordered_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_ordered_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_ordered_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_ordered_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_ordered_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_ordered_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_C(fh, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_ordered_begin_c wrapper function 
******************************************************/
int MPI_File_read_ordered_begin_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_ordered_begin_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_ordered_begin_c(fh, buf, count, datatype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_ordered_begin_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_ORDERED_BEGIN_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{

  *ierr = MPI_File_read_ordered_begin_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype);
  return ;
}

/******************************************************
***      MPI_File_read_ordered_begin_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_ordered_begin_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_ordered_begin_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_ordered_begin_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_ordered_begin_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_ordered_begin_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}


/******************************************************
***      MPI_File_read_shared_c wrapper function 
******************************************************/
int MPI_File_read_shared_c(MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_read_shared_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_read_shared_c(fh, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_read_shared_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_READ_SHARED_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_File_read_shared_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_read_shared_c wrapper function (lowercase)
******************************************************/
void mpi_file_read_shared_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_SHARED_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_shared_c wrapper function (lowercase_)
******************************************************/
void mpi_file_read_shared_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_SHARED_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_read_shared_c wrapper function (lowercase__)
******************************************************/
void mpi_file_read_shared_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_READ_SHARED_C(fh, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_c wrapper function 
******************************************************/
int MPI_File_write_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_c(fh, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_File_write_c(/* MPI_HANDLE_TYPES */ fh, buf, count, /* MPI_HANDLE_TYPES */ datatype,
    status);
  return ;
}

/******************************************************
***      MPI_File_write_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_C(fh, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_all_c wrapper function 
******************************************************/
int MPI_File_write_all_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_all_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_all_c(fh, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_all_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_ALL_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_File_write_all_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_write_all_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_all_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_all_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_all_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_all_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_all_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_C(fh, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_all_begin_c wrapper function 
******************************************************/
int MPI_File_write_all_begin_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_all_begin_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_all_begin_c(fh, buf, count, datatype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_all_begin_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_ALL_BEGIN_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{

  *ierr = MPI_File_write_all_begin_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype);
  return ;
}

/******************************************************
***      MPI_File_write_all_begin_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_all_begin_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_all_begin_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_all_begin_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_all_begin_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_all_begin_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_at_c wrapper function 
******************************************************/
int MPI_File_write_at_c(MPI_File fh, MPI_Offset offset, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype,
MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_at_c(fh, offset, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_at_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_AT_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{

  *ierr = MPI_File_write_at_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_write_at_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_at_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_at_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_at_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_at_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_at_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_at_all_c wrapper function 
******************************************************/
int MPI_File_write_at_all_c(MPI_File fh, MPI_Offset offset, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype,
MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_all_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_at_all_c(fh, offset, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_at_all_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_AT_ALL_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{

  *ierr = MPI_File_write_at_all_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_write_at_all_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_at_all_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_at_all_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_at_all_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_at_all_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_at_all_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_C(fh, offset, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_at_all_begin_c wrapper function 
******************************************************/
int MPI_File_write_at_all_begin_c(MPI_File fh, MPI_Offset offset, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_all_begin_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_at_all_begin_c(fh, offset, buf, count, datatype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_at_all_begin_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_AT_ALL_BEGIN_C(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{

  *ierr = MPI_File_write_at_all_begin_c(/* MPI_HANDLE_TYPES */ fh, offset, buf, count,
    /* MPI_HANDLE_TYPES */ datatype);
  return ;
}

/******************************************************
***      MPI_File_write_at_all_begin_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_at_all_begin_c(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_BEGIN_C(fh, offset, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_at_all_begin_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_at_all_begin_c_(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_BEGIN_C(fh, offset, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_at_all_begin_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_at_all_begin_c__(MPI_Fint fh, MPI_Offset offset, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_BEGIN_C(fh, offset, buf, count, datatype, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_ordered_c wrapper function 
******************************************************/
int MPI_File_write_ordered_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_ordered_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_ordered_c(fh, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_ordered_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_ORDERED_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_File_write_ordered_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_write_ordered_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_ordered_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_ordered_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_ordered_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_ordered_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_ordered_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_C(fh, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_ordered_begin_c wrapper function 
******************************************************/
int MPI_File_write_ordered_begin_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_ordered_begin_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_ordered_begin_c(fh, buf, count, datatype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_ordered_begin_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_ORDERED_BEGIN_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{

  *ierr = MPI_File_write_ordered_begin_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype);
  return ;
}

/******************************************************
***      MPI_File_write_ordered_begin_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_ordered_begin_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_ordered_begin_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_ordered_begin_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_ordered_begin_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_ordered_begin_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_BEGIN_C(fh, buf, count, datatype, ierr);
  return ;
}


/******************************************************
***      MPI_File_write_shared_c wrapper function 
******************************************************/
int MPI_File_write_shared_c(MPI_File fh, TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_File_write_shared_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_File_write_shared_c(fh, buf, count, datatype, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_File_write_shared_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_FILE_WRITE_SHARED_C(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_File_write_shared_c(/* MPI_HANDLE_TYPES */ fh, buf, count,
    /* MPI_HANDLE_TYPES */ datatype, status);
  return ;
}

/******************************************************
***      MPI_File_write_shared_c wrapper function (lowercase)
******************************************************/
void mpi_file_write_shared_c(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_SHARED_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_shared_c wrapper function (lowercase_)
******************************************************/
void mpi_file_write_shared_c_(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_SHARED_C(fh, buf, count, datatype, status, ierr);
  return ;
}

/******************************************************
***      MPI_File_write_shared_c wrapper function (lowercase__)
******************************************************/
void mpi_file_write_shared_c__(MPI_Fint fh, MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_SHARED_C(fh, buf, count, datatype, status, ierr);
  return ;
}


/******************************************************
***      MPI_Gather_c wrapper function 
******************************************************/
int MPI_Gather_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Gather_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Gather_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Gather_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_GATHER_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Gather_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Gather_c wrapper function (lowercase)
******************************************************/
void mpi_gather_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_GATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Gather_c wrapper function (lowercase_)
******************************************************/
void mpi_gather_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_GATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Gather_c wrapper function (lowercase__)
******************************************************/
void mpi_gather_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_GATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Igather_c wrapper function 
******************************************************/
int MPI_Igather_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Igather_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Igather_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Igather_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IGATHER_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Igather_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm,
    request);
  return ;
}

/******************************************************
***      MPI_Igather_c wrapper function (lowercase)
******************************************************/
void mpi_igather_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Igather_c wrapper function (lowercase_)
******************************************************/
void mpi_igather_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Igather_c wrapper function (lowercase__)
******************************************************/
void mpi_igather_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request,
    ierr);
  return ;
}


/******************************************************
***      MPI_Gather_init wrapper function 
******************************************************/
int MPI_Gather_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Gather_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Gather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Gather_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_GATHER_INIT(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{

  *ierr = MPI_Gather_init(sendbuf, /* MPI_HANDLE_TYPES */ sendcount,
    /* MPI_HANDLE_TYPES */ sendtype, recvbuf, /* MPI_HANDLE_TYPES */ recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Gather_init wrapper function (lowercase)
******************************************************/
void mpi_gather_init(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_GATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Gather_init wrapper function (lowercase_)
******************************************************/
void mpi_gather_init_(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_GATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Gather_init wrapper function (lowercase__)
******************************************************/
void mpi_gather_init__(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_GATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Gather_init_c wrapper function 
******************************************************/
int MPI_Gather_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Gather_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Gather_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Gather_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_GATHER_INIT_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{

  *ierr = MPI_Gather_init_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Gather_init_c wrapper function (lowercase)
******************************************************/
void mpi_gather_init_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_GATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Gather_init_c wrapper function (lowercase_)
******************************************************/
void mpi_gather_init_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_GATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Gather_init_c wrapper function (lowercase__)
******************************************************/
void mpi_gather_init_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_GATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Gatherv_c wrapper function 
******************************************************/
int MPI_Gatherv_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype recvtype, int root,
MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Gatherv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Gatherv_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Gatherv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_GATHERV_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Gatherv_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcounts,
    displs, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Gatherv_c wrapper function (lowercase)
******************************************************/
void mpi_gatherv_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_GATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    ierr);
  return ;
}

/******************************************************
***      MPI_Gatherv_c wrapper function (lowercase_)
******************************************************/
void mpi_gatherv_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_GATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    ierr);
  return ;
}

/******************************************************
***      MPI_Gatherv_c wrapper function (lowercase__)
******************************************************/
void mpi_gatherv_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_GATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    ierr);
  return ;
}


/******************************************************
***      MPI_Igatherv_c wrapper function 
******************************************************/
int MPI_Igatherv_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype recvtype, int root,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Igatherv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Igatherv_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Igatherv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IGATHERV_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Igatherv_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcounts,
    displs, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Igatherv_c wrapper function (lowercase)
******************************************************/
void mpi_igatherv_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Igatherv_c wrapper function (lowercase_)
******************************************************/
void mpi_igatherv_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Igatherv_c wrapper function (lowercase__)
******************************************************/
void mpi_igatherv_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Gatherv_init wrapper function 
******************************************************/
int MPI_Gatherv_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, TAU_MPICH3_CONST int recvcounts[],
TAU_MPICH3_CONST int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Gatherv_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Gatherv_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Gatherv_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_GATHERV_INIT(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, int recvcounts[],
int displs[], MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Gatherv_init(sendbuf, /* MPI_HANDLE_TYPES */ sendcount,
    /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcounts, displs, /* MPI_HANDLE_TYPES */ recvtype,
    /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Gatherv_init wrapper function (lowercase)
******************************************************/
void mpi_gatherv_init(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, int recvcounts[],
int displs[], MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_GATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Gatherv_init wrapper function (lowercase_)
******************************************************/
void mpi_gatherv_init_(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, int recvcounts[],
int displs[], MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_GATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Gatherv_init wrapper function (lowercase__)
******************************************************/
void mpi_gatherv_init__(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, int recvcounts[],
int displs[], MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_GATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm,
    info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Gatherv_init_c wrapper function 
******************************************************/
int MPI_Gatherv_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype recvtype, int root,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Gatherv_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Gatherv_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root,
    comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Gatherv_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_GATHERV_INIT_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Gatherv_init_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcounts, displs, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Gatherv_init_c wrapper function (lowercase)
******************************************************/
void mpi_gatherv_init_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_GATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root,
    comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Gatherv_init_c wrapper function (lowercase_)
******************************************************/
void mpi_gatherv_init_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_GATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root,
    comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Gatherv_init_c wrapper function (lowercase__)
******************************************************/
void mpi_gatherv_init_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_GATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root,
    comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Get_c wrapper function 
******************************************************/
int MPI_Get_c(void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype target_datatype, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Get_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Get_c(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Get_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_GET_C(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint * ierr)
{

  *ierr = MPI_Get_c(origin_addr, origin_count, /* MPI_HANDLE_TYPES */ origin_datatype,
    /* MPI_HANDLE_TYPES */ target_rank, target_disp, target_count,
    /* MPI_HANDLE_TYPES */ target_datatype, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Get_c wrapper function (lowercase)
******************************************************/
void mpi_get_c(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_GET_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, ierr);
  return ;
}

/******************************************************
***      MPI_Get_c wrapper function (lowercase_)
******************************************************/
void mpi_get_c_(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_GET_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, ierr);
  return ;
}

/******************************************************
***      MPI_Get_c wrapper function (lowercase__)
******************************************************/
void mpi_get_c__(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_GET_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, ierr);
  return ;
}


/******************************************************
***      MPI_Rget_c wrapper function 
******************************************************/
int MPI_Rget_c(void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype target_datatype, MPI_Win win,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Rget_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Rget_c(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Rget_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_RGET_C(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Rget_c(origin_addr, origin_count, /* MPI_HANDLE_TYPES */ origin_datatype,
    /* MPI_HANDLE_TYPES */ target_rank, target_disp, target_count,
    /* MPI_HANDLE_TYPES */ target_datatype, /* MPI_HANDLE_TYPES */ win, request);
  return ;
}

/******************************************************
***      MPI_Rget_c wrapper function (lowercase)
******************************************************/
void mpi_rget_c(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RGET_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rget_c wrapper function (lowercase_)
******************************************************/
void mpi_rget_c_(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RGET_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rget_c wrapper function (lowercase__)
******************************************************/
void mpi_rget_c__(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RGET_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}


/******************************************************
***      MPI_Get_accumulate_c wrapper function 
******************************************************/
int MPI_Get_accumulate_c(TAU_MPICH3_CONST void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype, void* result_addr,
MPI_Count result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Get_accumulate_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Get_accumulate_c(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Get_accumulate_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_GET_ACCUMULATE_C(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Count result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{

  *ierr = MPI_Get_accumulate_c(origin_addr, origin_count, /* MPI_HANDLE_TYPES */ origin_datatype,
    result_addr, result_count, /* MPI_HANDLE_TYPES */ result_datatype,
    /* MPI_HANDLE_TYPES */ target_rank, target_disp, target_count,
    /* MPI_HANDLE_TYPES */ target_datatype, /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Get_accumulate_c wrapper function (lowercase)
******************************************************/
void mpi_get_accumulate_c(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Count result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_GET_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
  return ;
}

/******************************************************
***      MPI_Get_accumulate_c wrapper function (lowercase_)
******************************************************/
void mpi_get_accumulate_c_(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Count result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_GET_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
  return ;
}

/******************************************************
***      MPI_Get_accumulate_c wrapper function (lowercase__)
******************************************************/
void mpi_get_accumulate_c__(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Count result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Fint * ierr)
{
  MPI_GET_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr);
  return ;
}


/******************************************************
***      MPI_Rget_accumulate_c wrapper function 
******************************************************/
int MPI_Rget_accumulate_c(TAU_MPICH3_CONST void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype, void* result_addr,
MPI_Count result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Rget_accumulate_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Rget_accumulate_c(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Rget_accumulate_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_RGET_ACCUMULATE_C(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Count result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Request* request,
MPI_Fint * ierr)
{

  *ierr = MPI_Rget_accumulate_c(origin_addr, origin_count, /* MPI_HANDLE_TYPES */ origin_datatype,
    result_addr, result_count, /* MPI_HANDLE_TYPES */ result_datatype,
    /* MPI_HANDLE_TYPES */ target_rank, target_disp, target_count,
    /* MPI_HANDLE_TYPES */ target_datatype, /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ win,
    request);
  return ;
}

/******************************************************
***      MPI_Rget_accumulate_c wrapper function (lowercase)
******************************************************/
void mpi_rget_accumulate_c(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Count result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_RGET_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rget_accumulate_c wrapper function (lowercase_)
******************************************************/
void mpi_rget_accumulate_c_(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Count result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_RGET_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rget_accumulate_c wrapper function (lowercase__)
******************************************************/
void mpi_rget_accumulate_c__(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Aint result_addr,
MPI_Count result_count, MPI_Fint result_datatype, MPI_Fint target_rank, MPI_Aint target_disp,
MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint op, MPI_Fint win, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_RGET_ACCUMULATE_C(origin_addr, origin_count, origin_datatype, result_addr, result_count,
    result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr);
  return ;
}


/******************************************************
***      MPI_Get_count_c wrapper function 
******************************************************/
int MPI_Get_count_c(TAU_MPICH3_CONST MPI_Status* status, MPI_Datatype datatype, MPI_Count* count)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Get_count_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Get_count_c(status, datatype, count);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Get_count_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_GET_COUNT_C(MPI_Status* status, MPI_Fint datatype, MPI_Count* count, MPI_Fint * ierr)
{

  *ierr = MPI_Get_count_c(status, /* MPI_HANDLE_TYPES */ datatype, count);
  return ;
}

/******************************************************
***      MPI_Get_count_c wrapper function (lowercase)
******************************************************/
void mpi_get_count_c(MPI_Status* status, MPI_Fint datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_COUNT_C(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Get_count_c wrapper function (lowercase_)
******************************************************/
void mpi_get_count_c_(MPI_Status* status, MPI_Fint datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_COUNT_C(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Get_count_c wrapper function (lowercase__)
******************************************************/
void mpi_get_count_c__(MPI_Status* status, MPI_Fint datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_COUNT_C(status, datatype, count, ierr);
  return ;
}


//New version of MPI_Get_elements_x
/******************************************************
***      MPI_Get_elements_c wrapper function 
******************************************************/
int MPI_Get_elements_c(TAU_MPICH3_CONST MPI_Status* status, MPI_Datatype datatype, MPI_Count* count)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Get_elements_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Get_elements_c(status, datatype, count);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Get_elements_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_GET_ELEMENTS_C(MPI_Status* status, MPI_Fint datatype, MPI_Count* count, MPI_Fint * ierr)
{

  *ierr = MPI_Get_elements_c(status, /* MPI_HANDLE_TYPES */ datatype, count);
  return ;
}

/******************************************************
***      MPI_Get_elements_c wrapper function (lowercase)
******************************************************/
void mpi_get_elements_c(MPI_Status* status, MPI_Fint datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_ELEMENTS_C(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Get_elements_c wrapper function (lowercase_)
******************************************************/
void mpi_get_elements_c_(MPI_Status* status, MPI_Fint datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_ELEMENTS_C(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Get_elements_c wrapper function (lowercase__)
******************************************************/
void mpi_get_elements_c__(MPI_Status* status, MPI_Fint datatype, MPI_Count* count, MPI_Fint * ierr)
{
  MPI_GET_ELEMENTS_C(status, datatype, count, ierr);
  return ;
}


/******************************************************
***      MPI_Ibsend_c wrapper function 
******************************************************/
int MPI_Ibsend_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ibsend_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ibsend_c(buf, count, datatype, dest, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ibsend_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IBSEND_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ibsend_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ibsend_c wrapper function (lowercase)
******************************************************/
void mpi_ibsend_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IBSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ibsend_c wrapper function (lowercase_)
******************************************************/
void mpi_ibsend_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IBSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ibsend_c wrapper function (lowercase__)
******************************************************/
void mpi_ibsend_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IBSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Imrecv_c wrapper function 
******************************************************/
int MPI_Imrecv_c(void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Message* message, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Imrecv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Imrecv_c(buf, count, datatype, message, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Imrecv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IMRECV_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Message* message, MPI_Request* request,
MPI_Fint * ierr)
{

  *ierr = MPI_Imrecv_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, message, request);
  return ;
}

/******************************************************
***      MPI_Imrecv_c wrapper function (lowercase)
******************************************************/
void mpi_imrecv_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Message* message, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_IMRECV_C(buf, count, datatype, message, request, ierr);
  return ;
}

/******************************************************
***      MPI_Imrecv_c wrapper function (lowercase_)
******************************************************/
void mpi_imrecv_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Message* message, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_IMRECV_C(buf, count, datatype, message, request, ierr);
  return ;
}

/******************************************************
***      MPI_Imrecv_c wrapper function (lowercase__)
******************************************************/
void mpi_imrecv_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Message* message, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_IMRECV_C(buf, count, datatype, message, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_allgather_c wrapper function 
******************************************************/
int MPI_Neighbor_allgather_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgather_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_allgather_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_allgather_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLGATHER_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_allgather_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_allgather_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_allgather_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_allgather_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_allgather_c wrapper function 
******************************************************/
int MPI_Ineighbor_allgather_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_allgather_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ineighbor_allgather_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_allgather_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_INEIGHBOR_ALLGATHER_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ineighbor_allgather_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgather_c wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_allgather_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgather_c wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_allgather_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgather_c wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_allgather_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_allgather_init wrapper function 
******************************************************/
int MPI_Neighbor_allgather_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgather_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_allgather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_allgather_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLGATHER_INIT(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_allgather_init(sendbuf, /* MPI_HANDLE_TYPES */ sendcount,
    /* MPI_HANDLE_TYPES */ sendtype, recvbuf, /* MPI_HANDLE_TYPES */ recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info,
    request);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_init wrapper function (lowercase)
******************************************************/
void mpi_neighbor_allgather_init(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_init wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_allgather_init_(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_init wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_allgather_init__(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_allgather_init_c wrapper function 
******************************************************/
int MPI_Neighbor_allgather_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgather_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_allgather_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_allgather_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLGATHER_INIT_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_allgather_init_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype,
    recvbuf, recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_init_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_allgather_init_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_init_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_allgather_init_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgather_init_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_allgather_init_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_allgatherv_c wrapper function 
******************************************************/
int MPI_Neighbor_allgatherv_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgatherv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_allgatherv_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_allgatherv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLGATHERV_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_allgatherv_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcounts, displs, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_allgatherv_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_allgatherv_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_allgatherv_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_allgatherv_c wrapper function 
******************************************************/
int MPI_Ineighbor_allgatherv_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype recvtype, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_allgatherv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ineighbor_allgatherv_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_allgatherv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_INEIGHBOR_ALLGATHERV_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ineighbor_allgatherv_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcounts, displs, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgatherv_c wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_allgatherv_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgatherv_c wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_allgatherv_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_allgatherv_c wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_allgatherv_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLGATHERV_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_allgatherv_init wrapper function 
******************************************************/
int MPI_Neighbor_allgatherv_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, TAU_MPICH3_CONST int recvcounts[],
TAU_MPICH3_CONST int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgatherv_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_allgatherv_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_allgatherv_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLGATHERV_INIT(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, int recvcounts[],
int displs[], MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_allgatherv_init(sendbuf, /* MPI_HANDLE_TYPES */ sendcount,
    /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcounts, displs, /* MPI_HANDLE_TYPES */ recvtype,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_init wrapper function (lowercase)
******************************************************/
void mpi_neighbor_allgatherv_init(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, int recvcounts[],
int displs[], MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_init wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_allgatherv_init_(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, int recvcounts[],
int displs[], MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_init wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_allgatherv_init__(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, int recvcounts[],
int displs[], MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
    comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_allgatherv_init_c wrapper function 
******************************************************/
int MPI_Neighbor_allgatherv_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype recvtype, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_allgatherv_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_allgatherv_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
    recvtype, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_allgatherv_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLGATHERV_INIT_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_allgatherv_init_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype,
    recvbuf, recvcounts, displs, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_init_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_allgatherv_init_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
    recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_init_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_allgatherv_init_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
    recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_allgatherv_init_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_allgatherv_init_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Count recvcounts[], MPI_Aint displs[], MPI_Fint recvtype, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLGATHERV_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
    recvtype, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoall_c wrapper function 
******************************************************/
int MPI_Neighbor_alltoall_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoall_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoall_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoall_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALL_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoall_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoall_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoall_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoall_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_alltoall_c wrapper function 
******************************************************/
int MPI_Ineighbor_alltoall_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoall_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ineighbor_alltoall_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_alltoall_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_INEIGHBOR_ALLTOALL_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ineighbor_alltoall_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoall_c wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_alltoall_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoall_c wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_alltoall_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoall_c wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_alltoall_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALL_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoall_init wrapper function 
******************************************************/
int MPI_Neighbor_alltoall_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoall_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoall_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoall_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALL_INIT(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoall_init(sendbuf, /* MPI_HANDLE_TYPES */ sendcount,
    /* MPI_HANDLE_TYPES */ sendtype, recvbuf, /* MPI_HANDLE_TYPES */ recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info,
    request);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_init wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoall_init(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_init wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoall_init_(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_init wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoall_init__(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, info,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoall_init_c wrapper function 
******************************************************/
int MPI_Neighbor_alltoall_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoall_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoall_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoall_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALL_INIT_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoall_init_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_init_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoall_init_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_init_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoall_init_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoall_init_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoall_init_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALL_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
    info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoallv_c wrapper function 
******************************************************/
int MPI_Neighbor_alltoallv_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[], MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint rdispls[], MPI_Datatype recvtype,
MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoallv_c(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoallv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALLV_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoallv_c(sendbuf, sendcounts, sdispls, /* MPI_HANDLE_TYPES */ sendtype,
    recvbuf, recvcounts, rdispls, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoallv_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoallv_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoallv_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_alltoallv_c wrapper function 
******************************************************/
int MPI_Ineighbor_alltoallv_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[], MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint rdispls[], MPI_Datatype recvtype,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoallv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ineighbor_alltoallv_c(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_alltoallv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_INEIGHBOR_ALLTOALLV_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ineighbor_alltoallv_c(sendbuf, sendcounts, sdispls, /* MPI_HANDLE_TYPES */ sendtype,
    recvbuf, recvcounts, rdispls, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm,
    request);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallv_c wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_alltoallv_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallv_c wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_alltoallv_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallv_c wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_alltoallv_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLV_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoallv_init wrapper function 
******************************************************/
int MPI_Neighbor_alltoallv_init(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int sendcounts[], TAU_MPICH3_CONST int sdispls[], MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST int recvcounts[], TAU_MPICH3_CONST int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallv_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoallv_init(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoallv_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALLV_INIT(MPI_Aint sendbuf, int sendcounts[], int sdispls[], MPI_Fint sendtype, MPI_Aint recvbuf,
int recvcounts[], int rdispls[], MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoallv_init(sendbuf, sendcounts, sdispls, /* MPI_HANDLE_TYPES */ sendtype,
    recvbuf, recvcounts, rdispls, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_init wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoallv_init(MPI_Aint sendbuf, int sendcounts[], int sdispls[], MPI_Fint sendtype, MPI_Aint recvbuf,
int recvcounts[], int rdispls[], MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_INIT(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_init wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoallv_init_(MPI_Aint sendbuf, int sendcounts[], int sdispls[], MPI_Fint sendtype, MPI_Aint recvbuf,
int recvcounts[], int rdispls[], MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_INIT(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_init wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoallv_init__(MPI_Aint sendbuf, int sendcounts[], int sdispls[], MPI_Fint sendtype, MPI_Aint recvbuf,
int recvcounts[], int rdispls[], MPI_Fint recvtype, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_INIT(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
    recvtype, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoallv_init_c wrapper function 
******************************************************/
int MPI_Neighbor_alltoallv_init_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[], MPI_Datatype sendtype,
void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[], TAU_MPICH3_CONST MPI_Aint rdispls[], MPI_Datatype recvtype,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallv_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoallv_init_c(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
    rdispls, recvtype, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoallv_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALLV_INIT_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoallv_init_c(sendbuf, sendcounts, sdispls,
    /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcounts, rdispls, /* MPI_HANDLE_TYPES */ recvtype,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_init_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoallv_init_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_INIT_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
    rdispls, recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_init_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoallv_init_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_INIT_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
    rdispls, recvtype, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallv_init_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoallv_init_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Aint rdispls[], MPI_Fint recvtype,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLV_INIT_C(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
    rdispls, recvtype, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoallw_c wrapper function 
******************************************************/
int MPI_Neighbor_alltoallw_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[],
TAU_MPICH3_CONST MPI_Datatype sendtypes[], void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[],
TAU_MPICH3_CONST MPI_Aint rdispls[], TAU_MPICH3_CONST MPI_Datatype recvtypes[], MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallw_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoallw_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoallw_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALLW_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoallw_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoallw_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoallw_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoallw_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ineighbor_alltoallw_c wrapper function 
******************************************************/
int MPI_Ineighbor_alltoallw_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[],
TAU_MPICH3_CONST MPI_Datatype sendtypes[], void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[],
TAU_MPICH3_CONST MPI_Aint rdispls[], TAU_MPICH3_CONST MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ineighbor_alltoallw_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ineighbor_alltoallw_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ineighbor_alltoallw_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_INEIGHBOR_ALLTOALLW_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Request* request,
MPI_Fint * ierr)
{

  *ierr = MPI_Ineighbor_alltoallw_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallw_c wrapper function (lowercase)
******************************************************/
void mpi_ineighbor_alltoallw_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallw_c wrapper function (lowercase_)
******************************************************/
void mpi_ineighbor_alltoallw_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ineighbor_alltoallw_c wrapper function (lowercase__)
******************************************************/
void mpi_ineighbor_alltoallw_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_INEIGHBOR_ALLTOALLW_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoallw_init wrapper function 
******************************************************/
int MPI_Neighbor_alltoallw_init(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[],
TAU_MPICH3_CONST MPI_Datatype sendtypes[], void* recvbuf, TAU_MPICH3_CONST int recvcounts[], TAU_MPICH3_CONST MPI_Aint rdispls[],
TAU_MPICH3_CONST MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallw_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoallw_init(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoallw_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALLW_INIT(MPI_Aint sendbuf, int sendcounts[], MPI_Aint sdispls[], MPI_Datatype sendtypes[],
MPI_Aint recvbuf, int recvcounts[], MPI_Aint rdispls[], MPI_Datatype recvtypes[],
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoallw_init(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_init wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoallw_init(MPI_Aint sendbuf, int sendcounts[], MPI_Aint sdispls[],  MPI_Datatype sendtypes[],
MPI_Aint recvbuf, int recvcounts[], MPI_Aint rdispls[], MPI_Datatype recvtypes[],
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_INIT(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_init wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoallw_init_(MPI_Aint sendbuf, int sendcounts[], MPI_Aint sdispls[], MPI_Datatype sendtypes[],
MPI_Aint recvbuf, int recvcounts[], MPI_Aint rdispls[], MPI_Datatype recvtypes[],
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_INIT(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_init wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoallw_init__(MPI_Aint sendbuf, int sendcounts[], MPI_Aint sdispls[], MPI_Datatype sendtypes[],
MPI_Aint recvbuf, int recvcounts[], MPI_Aint rdispls[], MPI_Datatype recvtypes[],
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_INIT(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
    recvtypes, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Neighbor_alltoallw_init_c wrapper function 
******************************************************/
int MPI_Neighbor_alltoallw_init_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint sdispls[],
TAU_MPICH3_CONST MPI_Datatype sendtypes[], void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[],
TAU_MPICH3_CONST MPI_Aint rdispls[], TAU_MPICH3_CONST MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Info info,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Neighbor_alltoallw_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Neighbor_alltoallw_init_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Neighbor_alltoallw_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_NEIGHBOR_ALLTOALLW_INIT_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Neighbor_alltoallw_init_c(sendbuf, sendcounts, sdispls, sendtypes, recvbuf,
    recvcounts, rdispls, recvtypes, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info,
    request);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_init_c wrapper function (lowercase)
******************************************************/
void mpi_neighbor_alltoallw_init_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_INIT_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_init_c wrapper function (lowercase_)
******************************************************/
void mpi_neighbor_alltoallw_init_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_INIT_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Neighbor_alltoallw_init_c wrapper function (lowercase__)
******************************************************/
void mpi_neighbor_alltoallw_init_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint sdispls[],
MPI_Datatype sendtypes[], MPI_Aint recvbuf, MPI_Count recvcounts[],
MPI_Aint rdispls[], MPI_Datatype recvtypes[], MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_NEIGHBOR_ALLTOALLW_INIT_C(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
    rdispls, recvtypes, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Irecv_c wrapper function 
******************************************************/
int MPI_Irecv_c(void* buf, MPI_Count count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Irecv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Irecv_c(buf, count, datatype, source, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Irecv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IRECV_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Irecv_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ source,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Irecv_c wrapper function (lowercase)
******************************************************/
void mpi_irecv_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IRECV_C(buf, count, datatype, source, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Irecv_c wrapper function (lowercase_)
******************************************************/
void mpi_irecv_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IRECV_C(buf, count, datatype, source, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Irecv_c wrapper function (lowercase__)
******************************************************/
void mpi_irecv_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IRECV_C(buf, count, datatype, source, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_c wrapper function 
******************************************************/
int MPI_Reduce_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, int root,
MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_c(sendbuf, recvbuf, count, datatype, op, root, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Reduce_c wrapper function (lowercase)
******************************************************/
void mpi_reduce_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_C(sendbuf, recvbuf, count, datatype, op, root, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_c wrapper function (lowercase_)
******************************************************/
void mpi_reduce_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_C(sendbuf, recvbuf, count, datatype, op, root, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_c wrapper function (lowercase__)
******************************************************/
void mpi_reduce_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_C(sendbuf, recvbuf, count, datatype, op, root, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ireduce_c wrapper function 
******************************************************/
int MPI_Ireduce_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, int root,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ireduce_c(sendbuf, recvbuf, count, datatype, op, root, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ireduce_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IREDUCE_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ireduce_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ireduce_c wrapper function (lowercase)
******************************************************/
void mpi_ireduce_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_C(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_c wrapper function (lowercase_)
******************************************************/
void mpi_ireduce_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_C(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_c wrapper function (lowercase__)
******************************************************/
void mpi_ireduce_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_C(sendbuf, recvbuf, count, datatype, op, root, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_init wrapper function 
******************************************************/
int MPI_Reduce_init(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_init(sendbuf, recvbuf, count, datatype, op, root, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_INIT(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_init(sendbuf, recvbuf, /* MPI_HANDLE_TYPES */ count,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Reduce_init wrapper function (lowercase)
******************************************************/
void mpi_reduce_init(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_INIT(sendbuf, recvbuf, count, datatype, op, root, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_init wrapper function (lowercase_)
******************************************************/
void mpi_reduce_init_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_INIT(sendbuf, recvbuf, count, datatype, op, root, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_init wrapper function (lowercase__)
******************************************************/
void mpi_reduce_init__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_INIT(sendbuf, recvbuf, count, datatype, op, root, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_init_c wrapper function 
******************************************************/
int MPI_Reduce_init_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, int root,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_init_c(sendbuf, recvbuf, count, datatype, op, root, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_INIT_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_init_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Reduce_init_c wrapper function (lowercase)
******************************************************/
void mpi_reduce_init_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_INIT_C(sendbuf, recvbuf, count, datatype, op, root, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_init_c wrapper function (lowercase_)
******************************************************/
void mpi_reduce_init_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_INIT_C(sendbuf, recvbuf, count, datatype, op, root, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_init_c wrapper function (lowercase__)
******************************************************/
void mpi_reduce_init_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint root,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_INIT_C(sendbuf, recvbuf, count, datatype, op, root, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_scatter_c wrapper function 
******************************************************/
int MPI_Reduce_scatter_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[], MPI_Datatype datatype, MPI_Op op,
MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_scatter_c(sendbuf, recvbuf, recvcounts, datatype, op, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_scatter_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_SCATTER_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_scatter_c(sendbuf, recvbuf, recvcounts, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_c wrapper function (lowercase)
******************************************************/
void mpi_reduce_scatter_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_c wrapper function (lowercase_)
******************************************************/
void mpi_reduce_scatter_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_c wrapper function (lowercase__)
******************************************************/
void mpi_reduce_scatter_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ireduce_scatter_c wrapper function 
******************************************************/
int MPI_Ireduce_scatter_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[], MPI_Datatype datatype, MPI_Op op,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_scatter_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ireduce_scatter_c(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ireduce_scatter_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IREDUCE_SCATTER_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ireduce_scatter_c(sendbuf, recvbuf, recvcounts, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter_c wrapper function (lowercase)
******************************************************/
void mpi_ireduce_scatter_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter_c wrapper function (lowercase_)
******************************************************/
void mpi_ireduce_scatter_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter_c wrapper function (lowercase__)
******************************************************/
void mpi_ireduce_scatter_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_scatter_init wrapper function 
******************************************************/
int MPI_Reduce_scatter_init(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, TAU_MPICH3_CONST int recvcounts[], MPI_Datatype datatype, MPI_Op op,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_scatter_init(sendbuf, recvbuf, recvcounts, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_scatter_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_SCATTER_INIT(MPI_Aint sendbuf, MPI_Aint recvbuf, int recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_scatter_init(sendbuf, recvbuf, recvcounts, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_init wrapper function (lowercase)
******************************************************/
void mpi_reduce_scatter_init(MPI_Aint sendbuf, MPI_Aint recvbuf, int recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_INIT(sendbuf, recvbuf, recvcounts, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_init wrapper function (lowercase_)
******************************************************/
void mpi_reduce_scatter_init_(MPI_Aint sendbuf, MPI_Aint recvbuf, int recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_INIT(sendbuf, recvbuf, recvcounts, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_init wrapper function (lowercase__)
******************************************************/
void mpi_reduce_scatter_init__(MPI_Aint sendbuf, MPI_Aint recvbuf, int recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_INIT(sendbuf, recvbuf, recvcounts, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_scatter_init_c wrapper function 
******************************************************/
int MPI_Reduce_scatter_init_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, TAU_MPICH3_CONST MPI_Count recvcounts[], MPI_Datatype datatype, MPI_Op op,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_scatter_init_c(sendbuf, recvbuf, recvcounts, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_scatter_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_SCATTER_INIT_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_scatter_init_c(sendbuf, recvbuf, recvcounts, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_init_c wrapper function (lowercase)
******************************************************/
void mpi_reduce_scatter_init_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_INIT_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_init_c wrapper function (lowercase_)
******************************************************/
void mpi_reduce_scatter_init_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_INIT_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_init_c wrapper function (lowercase__)
******************************************************/
void mpi_reduce_scatter_init_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcounts[], MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_INIT_C(sendbuf, recvbuf, recvcounts, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_scatter_block_c wrapper function 
******************************************************/
int MPI_Reduce_scatter_block_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count recvcount, MPI_Datatype datatype, MPI_Op op,
MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_block_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_scatter_block_c(sendbuf, recvbuf, recvcount, datatype, op, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_scatter_block_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_SCATTER_BLOCK_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_scatter_block_c(sendbuf, recvbuf, recvcount, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_c wrapper function (lowercase)
******************************************************/
void mpi_reduce_scatter_block_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_C(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_c wrapper function (lowercase_)
******************************************************/
void mpi_reduce_scatter_block_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_C(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_c wrapper function (lowercase__)
******************************************************/
void mpi_reduce_scatter_block_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_C(sendbuf, recvbuf, recvcount, datatype, op, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ireduce_scatter_block_c wrapper function 
******************************************************/
int MPI_Ireduce_scatter_block_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count recvcount, MPI_Datatype datatype, MPI_Op op,
MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ireduce_scatter_block_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ireduce_scatter_block_c(sendbuf, recvbuf, recvcount, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ireduce_scatter_block_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IREDUCE_SCATTER_BLOCK_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ireduce_scatter_block_c(sendbuf, recvbuf, recvcount, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter_block_c wrapper function (lowercase)
******************************************************/
void mpi_ireduce_scatter_block_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_BLOCK_C(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter_block_c wrapper function (lowercase_)
******************************************************/
void mpi_ireduce_scatter_block_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_BLOCK_C(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ireduce_scatter_block_c wrapper function (lowercase__)
******************************************************/
void mpi_ireduce_scatter_block_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IREDUCE_SCATTER_BLOCK_C(sendbuf, recvbuf, recvcount, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_scatter_block_init wrapper function 
******************************************************/
int MPI_Reduce_scatter_block_init(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_block_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_scatter_block_init(sendbuf, recvbuf, recvcount, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_scatter_block_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_SCATTER_BLOCK_INIT(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_scatter_block_init(sendbuf, recvbuf, /* MPI_HANDLE_TYPES */ recvcount,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_init wrapper function (lowercase)
******************************************************/
void mpi_reduce_scatter_block_init(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_INIT(sendbuf, recvbuf, recvcount, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_init wrapper function (lowercase_)
******************************************************/
void mpi_reduce_scatter_block_init_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_INIT(sendbuf, recvbuf, recvcount, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_init wrapper function (lowercase__)
******************************************************/
void mpi_reduce_scatter_block_init__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_INIT(sendbuf, recvbuf, recvcount, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_scatter_block_init_c wrapper function 
******************************************************/
int MPI_Reduce_scatter_block_init_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count recvcount, MPI_Datatype datatype, MPI_Op op,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_scatter_block_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_scatter_block_init_c(sendbuf, recvbuf, recvcount, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_scatter_block_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_SCATTER_BLOCK_INIT_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_scatter_block_init_c(sendbuf, recvbuf, recvcount,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_init_c wrapper function (lowercase)
******************************************************/
void mpi_reduce_scatter_block_init_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_INIT_C(sendbuf, recvbuf, recvcount, datatype, op, comm, info, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_init_c wrapper function (lowercase_)
******************************************************/
void mpi_reduce_scatter_block_init_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_INIT_C(sendbuf, recvbuf, recvcount, datatype, op, comm, info, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_scatter_block_init_c wrapper function (lowercase__)
******************************************************/
void mpi_reduce_scatter_block_init_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint datatype, MPI_Fint op,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_REDUCE_SCATTER_BLOCK_INIT_C(sendbuf, recvbuf, recvcount, datatype, op, comm, info, request,
    ierr);
  return ;
}


/******************************************************
***      MPI_Irsend_c wrapper function 
******************************************************/
int MPI_Irsend_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Irsend_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Irsend_c(buf, count, datatype, dest, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Irsend_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_IRSEND_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Irsend_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Irsend_c wrapper function (lowercase)
******************************************************/
void mpi_irsend_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IRSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Irsend_c wrapper function (lowercase_)
******************************************************/
void mpi_irsend_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IRSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Irsend_c wrapper function (lowercase__)
******************************************************/
void mpi_irsend_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_IRSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Scan_c wrapper function 
******************************************************/
int MPI_Scan_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scan_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scan_c(sendbuf, recvbuf, count, datatype, op, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scan_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCAN_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Scan_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Scan_c wrapper function (lowercase)
******************************************************/
void mpi_scan_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SCAN_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Scan_c wrapper function (lowercase_)
******************************************************/
void mpi_scan_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SCAN_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Scan_c wrapper function (lowercase__)
******************************************************/
void mpi_scan_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SCAN_C(sendbuf, recvbuf, count, datatype, op, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Iscan_c wrapper function 
******************************************************/
int MPI_Iscan_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iscan_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Iscan_c(sendbuf, recvbuf, count, datatype, op, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iscan_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISCAN_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Iscan_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Iscan_c wrapper function (lowercase)
******************************************************/
void mpi_iscan_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCAN_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscan_c wrapper function (lowercase_)
******************************************************/
void mpi_iscan_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCAN_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscan_c wrapper function (lowercase__)
******************************************************/
void mpi_iscan_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCAN_C(sendbuf, recvbuf, count, datatype, op, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Scan_init wrapper function 
******************************************************/
int MPI_Scan_init(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scan_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scan_init(sendbuf, recvbuf, count, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scan_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCAN_INIT(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Scan_init(sendbuf, recvbuf, /* MPI_HANDLE_TYPES */ count,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Scan_init wrapper function (lowercase)
******************************************************/
void mpi_scan_init(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCAN_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Scan_init wrapper function (lowercase_)
******************************************************/
void mpi_scan_init_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCAN_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Scan_init wrapper function (lowercase__)
******************************************************/
void mpi_scan_init__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Fint count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCAN_INIT(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Scan_init_c wrapper function 
******************************************************/
int MPI_Scan_init_c(TAU_MPICH3_CONST void* sendbuf, void* recvbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scan_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scan_init_c(sendbuf, recvbuf, count, datatype, op, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scan_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCAN_INIT_C(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Scan_init_c(sendbuf, recvbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op, /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Scan_init_c wrapper function (lowercase)
******************************************************/
void mpi_scan_init_c(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCAN_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Scan_init_c wrapper function (lowercase_)
******************************************************/
void mpi_scan_init_c_(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCAN_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Scan_init_c wrapper function (lowercase__)
******************************************************/
void mpi_scan_init_c__(MPI_Aint sendbuf, MPI_Aint recvbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCAN_INIT_C(sendbuf, recvbuf, count, datatype, op, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Scatter_c wrapper function 
******************************************************/
int MPI_Scatter_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scatter_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scatter_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scatter_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCATTER_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Scatter_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Scatter_c wrapper function (lowercase)
******************************************************/
void mpi_scatter_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_SCATTER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Scatter_c wrapper function (lowercase_)
******************************************************/
void mpi_scatter_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_SCATTER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Scatter_c wrapper function (lowercase__)
******************************************************/
void mpi_scatter_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_SCATTER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Iscatter_c wrapper function 
******************************************************/
int MPI_Iscatter_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iscatter_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Iscatter_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iscatter_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISCATTER_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Iscatter_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf, recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm,
    request);
  return ;
}

/******************************************************
***      MPI_Iscatter_c wrapper function (lowercase)
******************************************************/
void mpi_iscatter_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCATTER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Iscatter_c wrapper function (lowercase_)
******************************************************/
void mpi_iscatter_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCATTER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request,
    ierr);
  return ;
}

/******************************************************
***      MPI_Iscatter_c wrapper function (lowercase__)
******************************************************/
void mpi_iscatter_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCATTER_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request,
    ierr);
  return ;
}


/******************************************************
***      MPI_Scatter_init wrapper function 
******************************************************/
int MPI_Scatter_init(TAU_MPICH3_CONST void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scatter_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scatter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scatter_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCATTER_INIT(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{

  *ierr = MPI_Scatter_init(sendbuf, /* MPI_HANDLE_TYPES */ sendcount,
    /* MPI_HANDLE_TYPES */ sendtype, recvbuf, /* MPI_HANDLE_TYPES */ recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root, /* MPI_HANDLE_TYPES */ comm,
    /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Scatter_init wrapper function (lowercase)
******************************************************/
void mpi_scatter_init(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_SCATTER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Scatter_init wrapper function (lowercase_)
******************************************************/
void mpi_scatter_init_(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_SCATTER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Scatter_init wrapper function (lowercase__)
******************************************************/
void mpi_scatter_init__(MPI_Aint sendbuf, MPI_Fint sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Fint recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_SCATTER_INIT(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Scatter_init_c wrapper function 
******************************************************/
int MPI_Scatter_init_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scatter_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scatter_init_c(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scatter_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCATTER_INIT_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{

  *ierr = MPI_Scatter_init_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Scatter_init_c wrapper function (lowercase)
******************************************************/
void mpi_scatter_init_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_SCATTER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Scatter_init_c wrapper function (lowercase_)
******************************************************/
void mpi_scatter_init_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_SCATTER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Scatter_init_c wrapper function (lowercase__)
******************************************************/
void mpi_scatter_init_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Aint recvbuf, MPI_Count recvcount,
MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info, MPI_Request* request,
MPI_Fint * ierr)
{
  MPI_SCATTER_INIT_C(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, info,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Scatterv_c wrapper function 
******************************************************/
int MPI_Scatterv_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype sendtype,
void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scatterv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scatterv_c(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scatterv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCATTERV_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Scatterv_c(sendbuf, sendcounts, displs, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Scatterv_c wrapper function (lowercase)
******************************************************/
void mpi_scatterv_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SCATTERV_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    ierr);
  return ;
}

/******************************************************
***      MPI_Scatterv_c wrapper function (lowercase_)
******************************************************/
void mpi_scatterv_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SCATTERV_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    ierr);
  return ;
}

/******************************************************
***      MPI_Scatterv_c wrapper function (lowercase__)
******************************************************/
void mpi_scatterv_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SCATTERV_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    ierr);
  return ;
}


/******************************************************
***      MPI_Iscatterv_c wrapper function 
******************************************************/
int MPI_Iscatterv_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype sendtype,
void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Iscatterv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Iscatterv_c(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Iscatterv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISCATTERV_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Iscatterv_c(sendbuf, sendcounts, displs, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Iscatterv_c wrapper function (lowercase)
******************************************************/
void mpi_iscatterv_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCATTERV_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscatterv_c wrapper function (lowercase_)
******************************************************/
void mpi_iscatterv_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCATTERV_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    request, ierr);
  return ;
}

/******************************************************
***      MPI_Iscatterv_c wrapper function (lowercase__)
******************************************************/
void mpi_iscatterv_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISCATTERV_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    request, ierr);
  return ;
}


/******************************************************
***      MPI_Scatterv_init wrapper function 
******************************************************/
int MPI_Scatterv_init(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST int sendcounts[], TAU_MPICH3_CONST int displs[], MPI_Datatype sendtype,
void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scatterv_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scatterv_init(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
    comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scatterv_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCATTERV_INIT(MPI_Aint sendbuf, int sendcounts[], int displs[], MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Fint recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Scatterv_init(sendbuf, sendcounts, displs, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    /* MPI_HANDLE_TYPES */ recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Scatterv_init wrapper function (lowercase)
******************************************************/
void mpi_scatterv_init(MPI_Aint sendbuf, int sendcounts[], int displs[], MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Fint recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCATTERV_INIT(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Scatterv_init wrapper function (lowercase_)
******************************************************/
void mpi_scatterv_init_(MPI_Aint sendbuf, int sendcounts[], int displs[], MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Fint recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCATTERV_INIT(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Scatterv_init wrapper function (lowercase__)
******************************************************/
void mpi_scatterv_init__(MPI_Aint sendbuf, int sendcounts[], int displs[], MPI_Fint sendtype, MPI_Aint recvbuf,
MPI_Fint recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm, MPI_Fint info,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCATTERV_INIT(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm,
    info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Scatterv_init_c wrapper function 
******************************************************/
int MPI_Scatterv_init_c(TAU_MPICH3_CONST void* sendbuf, TAU_MPICH3_CONST MPI_Count sendcounts[], TAU_MPICH3_CONST MPI_Aint displs[], MPI_Datatype sendtype,
void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Scatterv_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Scatterv_init_c(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
    comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Scatterv_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SCATTERV_INIT_C(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Scatterv_init_c(sendbuf, sendcounts, displs, /* MPI_HANDLE_TYPES */ sendtype, recvbuf,
    recvcount, /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ root,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Scatterv_init_c wrapper function (lowercase)
******************************************************/
void mpi_scatterv_init_c(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCATTERV_INIT_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
    comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Scatterv_init_c wrapper function (lowercase_)
******************************************************/
void mpi_scatterv_init_c_(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCATTERV_INIT_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
    comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Scatterv_init_c wrapper function (lowercase__)
******************************************************/
void mpi_scatterv_init_c__(MPI_Aint sendbuf, MPI_Count sendcounts[], MPI_Aint displs[], MPI_Fint sendtype,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint root, MPI_Fint comm,
MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SCATTERV_INIT_C(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
    comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Isend_c wrapper function 
******************************************************/
int MPI_Isend_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Isend_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Isend_c(buf, count, datatype, dest, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Isend_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISEND_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Isend_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Isend_c wrapper function (lowercase)
******************************************************/
void mpi_isend_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isend_c wrapper function (lowercase_)
******************************************************/
void mpi_isend_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Isend_c wrapper function (lowercase__)
******************************************************/
void mpi_isend_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Issend_c wrapper function 
******************************************************/
int MPI_Issend_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Issend_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Issend_c(buf, count, datatype, dest, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Issend_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_ISSEND_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Issend_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Issend_c wrapper function (lowercase)
******************************************************/
void mpi_issend_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Issend_c wrapper function (lowercase_)
******************************************************/
void mpi_issend_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Issend_c wrapper function (lowercase__)
******************************************************/
void mpi_issend_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_ISSEND_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Mrecv_c wrapper function 
******************************************************/
int MPI_Mrecv_c(void* buf, MPI_Count count, MPI_Datatype datatype, MPI_Message* message, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Mrecv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Mrecv_c(buf, count, datatype, message, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Mrecv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_MRECV_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Message* message, MPI_Status* status,
MPI_Fint * ierr)
{

  *ierr = MPI_Mrecv_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, message, status);
  return ;
}

/******************************************************
***      MPI_Mrecv_c wrapper function (lowercase)
******************************************************/
void mpi_mrecv_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Message* message, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_MRECV_C(buf, count, datatype, message, status, ierr);
  return ;
}

/******************************************************
***      MPI_Mrecv_c wrapper function (lowercase_)
******************************************************/
void mpi_mrecv_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Message* message, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_MRECV_C(buf, count, datatype, message, status, ierr);
  return ;
}

/******************************************************
***      MPI_Mrecv_c wrapper function (lowercase__)
******************************************************/
void mpi_mrecv_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Message* message, MPI_Status* status,
MPI_Fint * ierr)
{
  MPI_MRECV_C(buf, count, datatype, message, status, ierr);
  return ;
}


/******************************************************
***      MPI_Op_create_c wrapper function 
******************************************************/
int MPI_Op_create_c(MPI_User_function_c* user_fn, int commute, MPI_Op* op)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Op_create_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Op_create_c(user_fn, commute, op);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Op_create_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_OP_CREATE_C(MPI_User_function_c* user_fn, MPI_Fint commute, MPI_Op* op, MPI_Fint * ierr)
{

  *ierr = MPI_Op_create_c(user_fn, /* MPI_HANDLE_TYPES */ commute, op);
  return ;
}

/******************************************************
***      MPI_Op_create_c wrapper function (lowercase)
******************************************************/
void mpi_op_create_c(MPI_User_function_c* user_fn, MPI_Fint commute, MPI_Op* op, MPI_Fint * ierr)
{
  MPI_OP_CREATE_C(user_fn, commute, op, ierr);
  return ;
}

/******************************************************
***      MPI_Op_create_c wrapper function (lowercase_)
******************************************************/
void mpi_op_create_c_(MPI_User_function_c* user_fn, MPI_Fint commute, MPI_Op* op, MPI_Fint * ierr)
{
  MPI_OP_CREATE_C(user_fn, commute, op, ierr);
  return ;
}

/******************************************************
***      MPI_Op_create_c wrapper function (lowercase__)
******************************************************/
void mpi_op_create_c__(MPI_User_function_c* user_fn, MPI_Fint commute, MPI_Op* op, MPI_Fint * ierr)
{
  MPI_OP_CREATE_C(user_fn, commute, op, ierr);
  return ;
}


/******************************************************
***      MPI_Pack_c wrapper function 
******************************************************/
int MPI_Pack_c(TAU_MPICH3_CONST void* inbuf, MPI_Count incount, MPI_Datatype datatype, void* outbuf, MPI_Count outsize,
MPI_Count* position, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Pack_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Pack_c(inbuf, incount, datatype, outbuf, outsize, position, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Pack_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_PACK_C(MPI_Aint inbuf, MPI_Count incount, MPI_Fint datatype, MPI_Aint outbuf, MPI_Count outsize,
MPI_Count* position, MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Pack_c(inbuf, incount, /* MPI_HANDLE_TYPES */ datatype, outbuf, outsize, position,
    /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Pack_c wrapper function (lowercase)
******************************************************/
void mpi_pack_c(MPI_Aint inbuf, MPI_Count incount, MPI_Fint datatype, MPI_Aint outbuf, MPI_Count outsize,
MPI_Count* position, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_PACK_C(inbuf, incount, datatype, outbuf, outsize, position, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Pack_c wrapper function (lowercase_)
******************************************************/
void mpi_pack_c_(MPI_Aint inbuf, MPI_Count incount, MPI_Fint datatype, MPI_Aint outbuf, MPI_Count outsize,
MPI_Count* position, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_PACK_C(inbuf, incount, datatype, outbuf, outsize, position, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Pack_c wrapper function (lowercase__)
******************************************************/
void mpi_pack_c__(MPI_Aint inbuf, MPI_Count incount, MPI_Fint datatype, MPI_Aint outbuf, MPI_Count outsize,
MPI_Count* position, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_PACK_C(inbuf, incount, datatype, outbuf, outsize, position, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Pack_external_c wrapper function 
******************************************************/
int MPI_Pack_external_c(TAU_MPICH3_CONST char datarep[], TAU_MPICH3_CONST void* inbuf, MPI_Count incount, MPI_Datatype datatype, void* outbuf,
MPI_Count outsize, MPI_Count* position)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Pack_external_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Pack_external_c(datarep, inbuf, incount, datatype, outbuf, outsize, position);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Pack_external_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_PACK_EXTERNAL_C(char datarep[], MPI_Aint inbuf, MPI_Count incount, MPI_Fint datatype, MPI_Aint outbuf,
MPI_Count outsize, MPI_Count* position, MPI_Fint * ierr)
{

  *ierr = MPI_Pack_external_c(datarep, inbuf, incount, /* MPI_HANDLE_TYPES */ datatype, outbuf,
    outsize, position);
  return ;
}

/******************************************************
***      MPI_Pack_external_c wrapper function (lowercase)
******************************************************/
void mpi_pack_external_c(char datarep[], MPI_Aint inbuf, MPI_Count incount, MPI_Fint datatype, MPI_Aint outbuf,
MPI_Count outsize, MPI_Count* position, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_C(datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr);
  return ;
}

/******************************************************
***      MPI_Pack_external_c wrapper function (lowercase_)
******************************************************/
void mpi_pack_external_c_(char datarep[], MPI_Aint inbuf, MPI_Count incount, MPI_Fint datatype, MPI_Aint outbuf,
MPI_Count outsize, MPI_Count* position, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_C(datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr);
  return ;
}

/******************************************************
***      MPI_Pack_external_c wrapper function (lowercase__)
******************************************************/
void mpi_pack_external_c__(char datarep[], MPI_Aint inbuf, MPI_Count incount, MPI_Fint datatype, MPI_Aint outbuf,
MPI_Count outsize, MPI_Count* position, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_C(datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr);
  return ;
}


/******************************************************
***      MPI_Pack_external_size_c wrapper function 
******************************************************/
int MPI_Pack_external_size_c(TAU_MPICH3_CONST char datarep[], MPI_Count incount, MPI_Datatype datatype, MPI_Count* size)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Pack_external_size_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Pack_external_size_c(datarep, incount, datatype, size);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Pack_external_size_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_PACK_EXTERNAL_SIZE_C(char datarep[], MPI_Count incount, MPI_Fint datatype, MPI_Count* size, MPI_Fint * ierr)
{

  *ierr = MPI_Pack_external_size_c(datarep, incount, /* MPI_HANDLE_TYPES */ datatype, size);
  return ;
}

/******************************************************
***      MPI_Pack_external_size_c wrapper function (lowercase)
******************************************************/
void mpi_pack_external_size_c(char datarep[], MPI_Count incount, MPI_Fint datatype, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_SIZE_C(datarep, incount, datatype, size, ierr);
  return ;
}

/******************************************************
***      MPI_Pack_external_size_c wrapper function (lowercase_)
******************************************************/
void mpi_pack_external_size_c_(char datarep[], MPI_Count incount, MPI_Fint datatype, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_SIZE_C(datarep, incount, datatype, size, ierr);
  return ;
}

/******************************************************
***      MPI_Pack_external_size_c wrapper function (lowercase__)
******************************************************/
void mpi_pack_external_size_c__(char datarep[], MPI_Count incount, MPI_Fint datatype, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_SIZE_C(datarep, incount, datatype, size, ierr);
  return ;
}


/******************************************************
***      MPI_Pack_size_c wrapper function 
******************************************************/
int MPI_Pack_size_c(MPI_Count incount, MPI_Datatype datatype, MPI_Comm comm, MPI_Count* size)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Pack_size_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Pack_size_c(incount, datatype, comm, size);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Pack_size_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_PACK_SIZE_C(MPI_Count incount, MPI_Fint datatype, MPI_Fint comm, MPI_Count* size, MPI_Fint * ierr)
{

  *ierr = MPI_Pack_size_c(incount, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ comm,
    size);
  return ;
}

/******************************************************
***      MPI_Pack_size_c wrapper function (lowercase)
******************************************************/
void mpi_pack_size_c(MPI_Count incount, MPI_Fint datatype, MPI_Fint comm, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_PACK_SIZE_C(incount, datatype, comm, size, ierr);
  return ;
}

/******************************************************
***      MPI_Pack_size_c wrapper function (lowercase_)
******************************************************/
void mpi_pack_size_c_(MPI_Count incount, MPI_Fint datatype, MPI_Fint comm, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_PACK_SIZE_C(incount, datatype, comm, size, ierr);
  return ;
}

/******************************************************
***      MPI_Pack_size_c wrapper function (lowercase__)
******************************************************/
void mpi_pack_size_c__(MPI_Count incount, MPI_Fint datatype, MPI_Fint comm, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_PACK_SIZE_C(incount, datatype, comm, size, ierr);
  return ;
}


/******************************************************
***      MPI_Parrived wrapper function 
******************************************************/
int MPI_Parrived(MPI_Request request, int partition, int* flag)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Parrived()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Parrived(request, partition, flag);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Parrived wrapper function (uppercase Fortran)
******************************************************/
void MPI_PARRIVED(MPI_Fint request, MPI_Fint partition, int* flag, MPI_Fint * ierr)
{

  *ierr = MPI_Parrived(/* MPI_HANDLE_TYPES */ request, /* MPI_HANDLE_TYPES */ partition, flag);
  return ;
}

/******************************************************
***      MPI_Parrived wrapper function (lowercase)
******************************************************/
void mpi_parrived(MPI_Fint request, MPI_Fint partition, int* flag, MPI_Fint * ierr)
{
  MPI_PARRIVED(request, partition, flag, ierr);
  return ;
}

/******************************************************
***      MPI_Parrived wrapper function (lowercase_)
******************************************************/
void mpi_parrived_(MPI_Fint request, MPI_Fint partition, int* flag, MPI_Fint * ierr)
{
  MPI_PARRIVED(request, partition, flag, ierr);
  return ;
}

/******************************************************
***      MPI_Parrived wrapper function (lowercase__)
******************************************************/
void mpi_parrived__(MPI_Fint request, MPI_Fint partition, int* flag, MPI_Fint * ierr)
{
  MPI_PARRIVED(request, partition, flag, ierr);
  return ;
}


/******************************************************
***      MPI_Pready wrapper function 
******************************************************/
int MPI_Pready(int partition, MPI_Request request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Pready()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Pready(partition, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Pready wrapper function (uppercase Fortran)
******************************************************/
void MPI_PREADY(MPI_Fint partition, MPI_Fint request, MPI_Fint * ierr)
{

  *ierr = MPI_Pready(/* MPI_HANDLE_TYPES */ partition, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_Pready wrapper function (lowercase)
******************************************************/
void mpi_pready(MPI_Fint partition, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY(partition, request, ierr);
  return ;
}

/******************************************************
***      MPI_Pready wrapper function (lowercase_)
******************************************************/
void mpi_pready_(MPI_Fint partition, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY(partition, request, ierr);
  return ;
}

/******************************************************
***      MPI_Pready wrapper function (lowercase__)
******************************************************/
void mpi_pready__(MPI_Fint partition, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY(partition, request, ierr);
  return ;
}


/******************************************************
***      MPI_Pready_list wrapper function 
******************************************************/
int MPI_Pready_list(int length, TAU_MPICH3_CONST int array_of_partitions[], MPI_Request request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Pready_list()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Pready_list(length, array_of_partitions, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Pready_list wrapper function (uppercase Fortran)
******************************************************/
void MPI_PREADY_LIST(MPI_Fint length, int array_of_partitions[], MPI_Fint request, MPI_Fint * ierr)
{

  *ierr = MPI_Pready_list(/* MPI_HANDLE_TYPES */ length, array_of_partitions,
    /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_Pready_list wrapper function (lowercase)
******************************************************/
void mpi_pready_list(MPI_Fint length, int array_of_partitions[], MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY_LIST(length, array_of_partitions, request, ierr);
  return ;
}

/******************************************************
***      MPI_Pready_list wrapper function (lowercase_)
******************************************************/
void mpi_pready_list_(MPI_Fint length, int array_of_partitions[], MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY_LIST(length, array_of_partitions, request, ierr);
  return ;
}

/******************************************************
***      MPI_Pready_list wrapper function (lowercase__)
******************************************************/
void mpi_pready_list__(MPI_Fint length, int array_of_partitions[], MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY_LIST(length, array_of_partitions, request, ierr);
  return ;
}


/******************************************************
***      MPI_Pready_range wrapper function 
******************************************************/
int MPI_Pready_range(int partition_low, int partition_high, MPI_Request request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Pready_range()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Pready_range(partition_low, partition_high, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Pready_range wrapper function (uppercase Fortran)
******************************************************/
void MPI_PREADY_RANGE(MPI_Fint partition_low, MPI_Fint partition_high, MPI_Fint request, MPI_Fint * ierr)
{

  *ierr = MPI_Pready_range(/* MPI_HANDLE_TYPES */ partition_low,
    /* MPI_HANDLE_TYPES */ partition_high, /* MPI_HANDLE_TYPES */ request);
  return ;
}

/******************************************************
***      MPI_Pready_range wrapper function (lowercase)
******************************************************/
void mpi_pready_range(MPI_Fint partition_low, MPI_Fint partition_high, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY_RANGE(partition_low, partition_high, request, ierr);
  return ;
}

/******************************************************
***      MPI_Pready_range wrapper function (lowercase_)
******************************************************/
void mpi_pready_range_(MPI_Fint partition_low, MPI_Fint partition_high, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY_RANGE(partition_low, partition_high, request, ierr);
  return ;
}

/******************************************************
***      MPI_Pready_range wrapper function (lowercase__)
******************************************************/
void mpi_pready_range__(MPI_Fint partition_low, MPI_Fint partition_high, MPI_Fint request, MPI_Fint * ierr)
{
  MPI_PREADY_RANGE(partition_low, partition_high, request, ierr);
  return ;
}


/******************************************************
***      MPI_Precv_init wrapper function 
******************************************************/
int MPI_Precv_init(void* buf, int partitions, MPI_Count count, MPI_Datatype datatype, int source, int tag,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Precv_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Precv_init(buf, partitions, count, datatype, source, tag, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Precv_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_PRECV_INIT(MPI_Aint buf, MPI_Fint partitions, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Precv_init(buf, /* MPI_HANDLE_TYPES */ partitions, count,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ source, /* MPI_HANDLE_TYPES */ tag,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Precv_init wrapper function (lowercase)
******************************************************/
void mpi_precv_init(MPI_Aint buf, MPI_Fint partitions, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_PRECV_INIT(buf, partitions, count, datatype, source, tag, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Precv_init wrapper function (lowercase_)
******************************************************/
void mpi_precv_init_(MPI_Aint buf, MPI_Fint partitions, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_PRECV_INIT(buf, partitions, count, datatype, source, tag, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Precv_init wrapper function (lowercase__)
******************************************************/
void mpi_precv_init__(MPI_Aint buf, MPI_Fint partitions, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_PRECV_INIT(buf, partitions, count, datatype, source, tag, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Psend_init wrapper function 
******************************************************/
int MPI_Psend_init(TAU_MPICH3_CONST void* buf, int partitions, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
MPI_Comm comm, MPI_Info info, MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Psend_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Psend_init(buf, partitions, count, datatype, dest, tag, comm, info, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Psend_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_PSEND_INIT(MPI_Aint buf, MPI_Fint partitions, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Psend_init(buf, /* MPI_HANDLE_TYPES */ partitions, count,
    /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest, /* MPI_HANDLE_TYPES */ tag,
    /* MPI_HANDLE_TYPES */ comm, /* MPI_HANDLE_TYPES */ info, request);
  return ;
}

/******************************************************
***      MPI_Psend_init wrapper function (lowercase)
******************************************************/
void mpi_psend_init(MPI_Aint buf, MPI_Fint partitions, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_PSEND_INIT(buf, partitions, count, datatype, dest, tag, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Psend_init wrapper function (lowercase_)
******************************************************/
void mpi_psend_init_(MPI_Aint buf, MPI_Fint partitions, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_PSEND_INIT(buf, partitions, count, datatype, dest, tag, comm, info, request, ierr);
  return ;
}

/******************************************************
***      MPI_Psend_init wrapper function (lowercase__)
******************************************************/
void mpi_psend_init__(MPI_Aint buf, MPI_Fint partitions, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag,
MPI_Fint comm, MPI_Fint info, MPI_Request* request, MPI_Fint * ierr)
{
  MPI_PSEND_INIT(buf, partitions, count, datatype, dest, tag, comm, info, request, ierr);
  return ;
}


/******************************************************
***      MPI_Put_c wrapper function 
******************************************************/
int MPI_Put_c(TAU_MPICH3_CONST void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype target_datatype, MPI_Win win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Put_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Put_c(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Put_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_PUT_C(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint * ierr)
{

  *ierr = MPI_Put_c(origin_addr, origin_count, /* MPI_HANDLE_TYPES */ origin_datatype,
    /* MPI_HANDLE_TYPES */ target_rank, target_disp, target_count,
    /* MPI_HANDLE_TYPES */ target_datatype, /* MPI_HANDLE_TYPES */ win);
  return ;
}

/******************************************************
***      MPI_Put_c wrapper function (lowercase)
******************************************************/
void mpi_put_c(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_PUT_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, ierr);
  return ;
}

/******************************************************
***      MPI_Put_c wrapper function (lowercase_)
******************************************************/
void mpi_put_c_(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_PUT_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, ierr);
  return ;
}

/******************************************************
***      MPI_Put_c wrapper function (lowercase__)
******************************************************/
void mpi_put_c__(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Fint * ierr)
{
  MPI_PUT_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, ierr);
  return ;
}


/******************************************************
***      MPI_Rput_c wrapper function 
******************************************************/
int MPI_Rput_c(TAU_MPICH3_CONST void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype, int target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Datatype target_datatype, MPI_Win win,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Rput_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Rput_c(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Rput_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_RPUT_C(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Rput_c(origin_addr, origin_count, /* MPI_HANDLE_TYPES */ origin_datatype,
    /* MPI_HANDLE_TYPES */ target_rank, target_disp, target_count,
    /* MPI_HANDLE_TYPES */ target_datatype, /* MPI_HANDLE_TYPES */ win, request);
  return ;
}

/******************************************************
***      MPI_Rput_c wrapper function (lowercase)
******************************************************/
void mpi_rput_c(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RPUT_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rput_c wrapper function (lowercase_)
******************************************************/
void mpi_rput_c_(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RPUT_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rput_c wrapper function (lowercase__)
******************************************************/
void mpi_rput_c__(MPI_Aint origin_addr, MPI_Count origin_count, MPI_Fint origin_datatype, MPI_Fint target_rank,
MPI_Aint target_disp, MPI_Count target_count, MPI_Fint target_datatype, MPI_Fint win,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RPUT_C(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
    target_datatype, win, request, ierr);
  return ;
}


/******************************************************
***      MPI_Recv_c wrapper function 
******************************************************/
int MPI_Recv_c(void* buf, MPI_Count count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Recv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Recv_c(buf, count, datatype, source, tag, comm, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Recv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_RECV_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_Recv_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ source,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, status);
  return ;
}

/******************************************************
***      MPI_Recv_c wrapper function (lowercase)
******************************************************/
void mpi_recv_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Status* status, MPI_Fint * ierr)
{
  MPI_RECV_C(buf, count, datatype, source, tag, comm, status, ierr);
  return ;
}

/******************************************************
***      MPI_Recv_c wrapper function (lowercase_)
******************************************************/
void mpi_recv_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Status* status, MPI_Fint * ierr)
{
  MPI_RECV_C(buf, count, datatype, source, tag, comm, status, ierr);
  return ;
}

/******************************************************
***      MPI_Recv_c wrapper function (lowercase__)
******************************************************/
void mpi_recv_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Status* status, MPI_Fint * ierr)
{
  MPI_RECV_C(buf, count, datatype, source, tag, comm, status, ierr);
  return ;
}


/******************************************************
***      MPI_Recv_init_c wrapper function 
******************************************************/
int MPI_Recv_init_c(void* buf, MPI_Count count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Recv_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Recv_init_c(buf, count, datatype, source, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Recv_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_RECV_INIT_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Recv_init_c(buf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ source, /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Recv_init_c wrapper function (lowercase)
******************************************************/
void mpi_recv_init_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RECV_INIT_C(buf, count, datatype, source, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Recv_init_c wrapper function (lowercase_)
******************************************************/
void mpi_recv_init_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RECV_INIT_C(buf, count, datatype, source, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Recv_init_c wrapper function (lowercase__)
******************************************************/
void mpi_recv_init_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint source, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RECV_INIT_C(buf, count, datatype, source, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Reduce_local_c wrapper function 
******************************************************/
int MPI_Reduce_local_c(TAU_MPICH3_CONST void* inbuf, void* inoutbuf, MPI_Count count, MPI_Datatype datatype, MPI_Op op)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Reduce_local_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Reduce_local_c(inbuf, inoutbuf, count, datatype, op);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Reduce_local_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_REDUCE_LOCAL_C(MPI_Aint inbuf, MPI_Aint inoutbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint * ierr)
{

  *ierr = MPI_Reduce_local_c(inbuf, inoutbuf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ op);
  return ;
}

/******************************************************
***      MPI_Reduce_local_c wrapper function (lowercase)
******************************************************/
void mpi_reduce_local_c(MPI_Aint inbuf, MPI_Aint inoutbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint * ierr)
{
  MPI_REDUCE_LOCAL_C(inbuf, inoutbuf, count, datatype, op, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_local_c wrapper function (lowercase_)
******************************************************/
void mpi_reduce_local_c_(MPI_Aint inbuf, MPI_Aint inoutbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint * ierr)
{
  MPI_REDUCE_LOCAL_C(inbuf, inoutbuf, count, datatype, op, ierr);
  return ;
}

/******************************************************
***      MPI_Reduce_local_c wrapper function (lowercase__)
******************************************************/
void mpi_reduce_local_c__(MPI_Aint inbuf, MPI_Aint inoutbuf, MPI_Count count, MPI_Fint datatype, MPI_Fint op, MPI_Fint * ierr)
{
  MPI_REDUCE_LOCAL_C(inbuf, inoutbuf, count, datatype, op, ierr);
  return ;
}


/******************************************************
***      MPI_Register_datarep_c wrapper function 
******************************************************/
int MPI_Register_datarep_c(TAU_MPICH3_CONST char* datarep, MPI_Datarep_conversion_function_c* read_conversion_fn,
MPI_Datarep_conversion_function_c* write_conversion_fn,
MPI_Datarep_extent_function* dtype_file_extent_fn, void* extra_state)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Register_datarep_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Register_datarep_c(datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn,
    extra_state);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Register_datarep_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_REGISTER_DATAREP_C(char* datarep, MPI_Datarep_conversion_function_c* read_conversion_fn,
MPI_Datarep_conversion_function_c* write_conversion_fn,
MPI_Datarep_extent_function* dtype_file_extent_fn, MPI_Aint extra_state, MPI_Fint * ierr)
{

  *ierr = MPI_Register_datarep_c(datarep, read_conversion_fn, write_conversion_fn,
    dtype_file_extent_fn, extra_state);
  return ;
}

/******************************************************
***      MPI_Register_datarep_c wrapper function (lowercase)
******************************************************/
void mpi_register_datarep_c(char* datarep, MPI_Datarep_conversion_function_c* read_conversion_fn,
MPI_Datarep_conversion_function_c* write_conversion_fn,
MPI_Datarep_extent_function* dtype_file_extent_fn, MPI_Aint extra_state, MPI_Fint * ierr)
{
  MPI_REGISTER_DATAREP_C(datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn,
    extra_state, ierr);
  return ;
}

/******************************************************
***      MPI_Register_datarep_c wrapper function (lowercase_)
******************************************************/
void mpi_register_datarep_c_(char* datarep, MPI_Datarep_conversion_function_c* read_conversion_fn,
MPI_Datarep_conversion_function_c* write_conversion_fn,
MPI_Datarep_extent_function* dtype_file_extent_fn, MPI_Aint extra_state, MPI_Fint * ierr)
{
  MPI_REGISTER_DATAREP_C(datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn,
    extra_state, ierr);
  return ;
}

/******************************************************
***      MPI_Register_datarep_c wrapper function (lowercase__)
******************************************************/
void mpi_register_datarep_c__(char* datarep, MPI_Datarep_conversion_function_c* read_conversion_fn,
MPI_Datarep_conversion_function_c* write_conversion_fn,
MPI_Datarep_extent_function* dtype_file_extent_fn, MPI_Aint extra_state, MPI_Fint * ierr)
{
  MPI_REGISTER_DATAREP_C(datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn,
    extra_state, ierr);
  return ;
}


/******************************************************
***      MPI_Rsend_c wrapper function 
******************************************************/
int MPI_Rsend_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Rsend_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Rsend_c(buf, count, datatype, dest, tag, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Rsend_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_RSEND_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Rsend_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Rsend_c wrapper function (lowercase)
******************************************************/
void mpi_rsend_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_RSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Rsend_c wrapper function (lowercase_)
******************************************************/
void mpi_rsend_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_RSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Rsend_c wrapper function (lowercase__)
******************************************************/
void mpi_rsend_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_RSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Rsend_init_c wrapper function 
******************************************************/
int MPI_Rsend_init_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Rsend_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Rsend_init_c(buf, count, datatype, dest, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Rsend_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_RSEND_INIT_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Rsend_init_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Rsend_init_c wrapper function (lowercase)
******************************************************/
void mpi_rsend_init_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rsend_init_c wrapper function (lowercase_)
******************************************************/
void mpi_rsend_init_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Rsend_init_c wrapper function (lowercase__)
******************************************************/
void mpi_rsend_init_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_RSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Send_c wrapper function 
******************************************************/
int MPI_Send_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Send_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Send_c(buf, count, datatype, dest, tag, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Send_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SEND_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Send_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Send_c wrapper function (lowercase)
******************************************************/
void mpi_send_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Send_c wrapper function (lowercase_)
******************************************************/
void mpi_send_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Send_c wrapper function (lowercase__)
******************************************************/
void mpi_send_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Send_init_c wrapper function 
******************************************************/
int MPI_Send_init_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Send_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Send_init_c(buf, count, datatype, dest, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Send_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SEND_INIT_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Send_init_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Send_init_c wrapper function (lowercase)
******************************************************/
void mpi_send_init_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Send_init_c wrapper function (lowercase_)
******************************************************/
void mpi_send_init_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Send_init_c wrapper function (lowercase__)
******************************************************/
void mpi_send_init_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Sendrecv_c wrapper function 
******************************************************/
int MPI_Sendrecv_c(TAU_MPICH3_CONST void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, int dest, int sendtag,
void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Sendrecv_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Sendrecv_c(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Sendrecv_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SENDRECV_C(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Fint dest, MPI_Fint sendtag,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint source, MPI_Fint recvtag,
MPI_Fint comm, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_Sendrecv_c(sendbuf, sendcount, /* MPI_HANDLE_TYPES */ sendtype,
    /* MPI_HANDLE_TYPES */ dest, /* MPI_HANDLE_TYPES */ sendtag, recvbuf, recvcount,
    /* MPI_HANDLE_TYPES */ recvtype, /* MPI_HANDLE_TYPES */ source, /* MPI_HANDLE_TYPES */ recvtag,
    /* MPI_HANDLE_TYPES */ comm, status);
  return ;
}

/******************************************************
***      MPI_Sendrecv_c wrapper function (lowercase)
******************************************************/
void mpi_sendrecv_c(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Fint dest, MPI_Fint sendtag,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint source, MPI_Fint recvtag,
MPI_Fint comm, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_SENDRECV_C(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, status, ierr);
  return ;
}

/******************************************************
***      MPI_Sendrecv_c wrapper function (lowercase_)
******************************************************/
void mpi_sendrecv_c_(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Fint dest, MPI_Fint sendtag,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint source, MPI_Fint recvtag,
MPI_Fint comm, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_SENDRECV_C(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, status, ierr);
  return ;
}

/******************************************************
***      MPI_Sendrecv_c wrapper function (lowercase__)
******************************************************/
void mpi_sendrecv_c__(MPI_Aint sendbuf, MPI_Count sendcount, MPI_Fint sendtype, MPI_Fint dest, MPI_Fint sendtag,
MPI_Aint recvbuf, MPI_Count recvcount, MPI_Fint recvtype, MPI_Fint source, MPI_Fint recvtag,
MPI_Fint comm, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_SENDRECV_C(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
    recvtag, comm, status, ierr);
  return ;
}


/******************************************************
***      MPI_Sendrecv_replace_c wrapper function 
******************************************************/
int MPI_Sendrecv_replace_c(void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag,
MPI_Comm comm, MPI_Status* status)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Sendrecv_replace_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Sendrecv_replace_c(buf, count, datatype, dest, sendtag, source, recvtag, comm, status);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Sendrecv_replace_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SENDRECV_REPLACE_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint sendtag, MPI_Fint source,
MPI_Fint recvtag, MPI_Fint comm, MPI_Status* status, MPI_Fint * ierr)
{

  *ierr = MPI_Sendrecv_replace_c(buf, count, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ dest, /* MPI_HANDLE_TYPES */ sendtag, /* MPI_HANDLE_TYPES */ source,
    /* MPI_HANDLE_TYPES */ recvtag, /* MPI_HANDLE_TYPES */ comm, status);
  return ;
}

/******************************************************
***      MPI_Sendrecv_replace_c wrapper function (lowercase)
******************************************************/
void mpi_sendrecv_replace_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint sendtag, MPI_Fint source,
MPI_Fint recvtag, MPI_Fint comm, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_SENDRECV_REPLACE_C(buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr);
  return ;
}

/******************************************************
***      MPI_Sendrecv_replace_c wrapper function (lowercase_)
******************************************************/
void mpi_sendrecv_replace_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint sendtag, MPI_Fint source,
MPI_Fint recvtag, MPI_Fint comm, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_SENDRECV_REPLACE_C(buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr);
  return ;
}

/******************************************************
***      MPI_Sendrecv_replace_c wrapper function (lowercase__)
******************************************************/
void mpi_sendrecv_replace_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint sendtag, MPI_Fint source,
MPI_Fint recvtag, MPI_Fint comm, MPI_Status* status, MPI_Fint * ierr)
{
  MPI_SENDRECV_REPLACE_C(buf, count, datatype, dest, sendtag, source, recvtag, comm, status, ierr);
  return ;
}


/******************************************************
***      MPI_Session_call_errhandler wrapper function 
******************************************************/
int MPI_Session_call_errhandler(MPI_Session session, int errorcode)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_call_errhandler()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_call_errhandler(session, errorcode);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_call_errhandler wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_CALL_ERRHANDLER(MPI_Fint session, MPI_Fint errorcode, MPI_Fint * ierr)
{

  *ierr = MPI_Session_call_errhandler(/* MPI_HANDLE_TYPES */ session,
    /* MPI_HANDLE_TYPES */ errorcode);
  return ;
}

/******************************************************
***      MPI_Session_call_errhandler wrapper function (lowercase)
******************************************************/
void mpi_session_call_errhandler(MPI_Fint session, MPI_Fint errorcode, MPI_Fint * ierr)
{
  MPI_SESSION_CALL_ERRHANDLER(session, errorcode, ierr);
  return ;
}

/******************************************************
***      MPI_Session_call_errhandler wrapper function (lowercase_)
******************************************************/
void mpi_session_call_errhandler_(MPI_Fint session, MPI_Fint errorcode, MPI_Fint * ierr)
{
  MPI_SESSION_CALL_ERRHANDLER(session, errorcode, ierr);
  return ;
}

/******************************************************
***      MPI_Session_call_errhandler wrapper function (lowercase__)
******************************************************/
void mpi_session_call_errhandler__(MPI_Fint session, MPI_Fint errorcode, MPI_Fint * ierr)
{
  MPI_SESSION_CALL_ERRHANDLER(session, errorcode, ierr);
  return ;
}


/******************************************************
***      MPI_Session_create_errhandler wrapper function 
******************************************************/
int MPI_Session_create_errhandler(MPI_Session_errhandler_function* session_errhandler_fn, MPI_Errhandler* errhandler)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_create_errhandler()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_create_errhandler(session_errhandler_fn, errhandler);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_create_errhandler wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_CREATE_ERRHANDLER(MPI_Session_errhandler_function* session_errhandler_fn, MPI_Errhandler* errhandler, MPI_Fint * ierr)
{

  *ierr = MPI_Session_create_errhandler(session_errhandler_fn, errhandler);
  return ;
}

/******************************************************
***      MPI_Session_create_errhandler wrapper function (lowercase)
******************************************************/
void mpi_session_create_errhandler(MPI_Session_errhandler_function* session_errhandler_fn, MPI_Errhandler* errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_CREATE_ERRHANDLER(session_errhandler_fn, errhandler, ierr);
  return ;
}

/******************************************************
***      MPI_Session_create_errhandler wrapper function (lowercase_)
******************************************************/
void mpi_session_create_errhandler_(MPI_Session_errhandler_function* session_errhandler_fn, MPI_Errhandler* errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_CREATE_ERRHANDLER(session_errhandler_fn, errhandler, ierr);
  return ;
}

/******************************************************
***      MPI_Session_create_errhandler wrapper function (lowercase__)
******************************************************/
void mpi_session_create_errhandler__(MPI_Session_errhandler_function* session_errhandler_fn, MPI_Errhandler* errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_CREATE_ERRHANDLER(session_errhandler_fn, errhandler, ierr);
  return ;
}


/******************************************************
***      MPI_Session_finalize wrapper function 
******************************************************/
int MPI_Session_finalize(MPI_Session* session)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_finalize()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_finalize(session);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_finalize wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_FINALIZE(MPI_Session* session, MPI_Fint * ierr)
{

  *ierr = MPI_Session_finalize(session);
  return ;
}

/******************************************************
***      MPI_Session_finalize wrapper function (lowercase)
******************************************************/
void mpi_session_finalize(MPI_Session* session, MPI_Fint * ierr)
{
  MPI_SESSION_FINALIZE(session, ierr);
  return ;
}

/******************************************************
***      MPI_Session_finalize wrapper function (lowercase_)
******************************************************/
void mpi_session_finalize_(MPI_Session* session, MPI_Fint * ierr)
{
  MPI_SESSION_FINALIZE(session, ierr);
  return ;
}

/******************************************************
***      MPI_Session_finalize wrapper function (lowercase__)
******************************************************/
void mpi_session_finalize__(MPI_Session* session, MPI_Fint * ierr)
{
  MPI_SESSION_FINALIZE(session, ierr);
  return ;
}


/******************************************************
***      MPI_Session_get_errhandler wrapper function 
******************************************************/
int MPI_Session_get_errhandler(MPI_Session session, MPI_Errhandler* errhandler)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_get_errhandler()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_get_errhandler(session, errhandler);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_get_errhandler wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_GET_ERRHANDLER(MPI_Fint session, MPI_Errhandler* errhandler, MPI_Fint * ierr)
{

  *ierr = MPI_Session_get_errhandler(/* MPI_HANDLE_TYPES */ session, errhandler);
  return ;
}

/******************************************************
***      MPI_Session_get_errhandler wrapper function (lowercase)
******************************************************/
void mpi_session_get_errhandler(MPI_Fint session, MPI_Errhandler* errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_GET_ERRHANDLER(session, errhandler, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_errhandler wrapper function (lowercase_)
******************************************************/
void mpi_session_get_errhandler_(MPI_Fint session, MPI_Errhandler* errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_GET_ERRHANDLER(session, errhandler, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_errhandler wrapper function (lowercase__)
******************************************************/
void mpi_session_get_errhandler__(MPI_Fint session, MPI_Errhandler* errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_GET_ERRHANDLER(session, errhandler, ierr);
  return ;
}


/******************************************************
***      MPI_Session_get_info wrapper function 
******************************************************/
int MPI_Session_get_info(MPI_Session session, MPI_Info* info_used)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_get_info()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_get_info(session, info_used);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_get_info wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_GET_INFO(MPI_Fint session, MPI_Info* info_used, MPI_Fint * ierr)
{

  *ierr = MPI_Session_get_info(/* MPI_HANDLE_TYPES */ session, info_used);
  return ;
}

/******************************************************
***      MPI_Session_get_info wrapper function (lowercase)
******************************************************/
void mpi_session_get_info(MPI_Fint session, MPI_Info* info_used, MPI_Fint * ierr)
{
  MPI_SESSION_GET_INFO(session, info_used, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_info wrapper function (lowercase_)
******************************************************/
void mpi_session_get_info_(MPI_Fint session, MPI_Info* info_used, MPI_Fint * ierr)
{
  MPI_SESSION_GET_INFO(session, info_used, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_info wrapper function (lowercase__)
******************************************************/
void mpi_session_get_info__(MPI_Fint session, MPI_Info* info_used, MPI_Fint * ierr)
{
  MPI_SESSION_GET_INFO(session, info_used, ierr);
  return ;
}


/******************************************************
***      MPI_Session_get_nth_pset wrapper function 
******************************************************/
int MPI_Session_get_nth_pset(MPI_Session session, MPI_Info info, int n, int* pset_len, char* pset_name)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_get_nth_pset()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_get_nth_pset(session, info, n, pset_len, pset_name);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_get_nth_pset wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_GET_NTH_PSET(MPI_Fint session, MPI_Fint info, MPI_Fint n, int* pset_len, char* pset_name, MPI_Fint * ierr)
{

  *ierr = MPI_Session_get_nth_pset(/* MPI_HANDLE_TYPES */ session, /* MPI_HANDLE_TYPES */ info,
    /* MPI_HANDLE_TYPES */ n, pset_len, pset_name);
  return ;
}

/******************************************************
***      MPI_Session_get_nth_pset wrapper function (lowercase)
******************************************************/
void mpi_session_get_nth_pset(MPI_Fint session, MPI_Fint info, MPI_Fint n, int* pset_len, char* pset_name, MPI_Fint * ierr)
{
  MPI_SESSION_GET_NTH_PSET(session, info, n, pset_len, pset_name, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_nth_pset wrapper function (lowercase_)
******************************************************/
void mpi_session_get_nth_pset_(MPI_Fint session, MPI_Fint info, MPI_Fint n, int* pset_len, char* pset_name, MPI_Fint * ierr)
{
  MPI_SESSION_GET_NTH_PSET(session, info, n, pset_len, pset_name, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_nth_pset wrapper function (lowercase__)
******************************************************/
void mpi_session_get_nth_pset__(MPI_Fint session, MPI_Fint info, MPI_Fint n, int* pset_len, char* pset_name, MPI_Fint * ierr)
{
  MPI_SESSION_GET_NTH_PSET(session, info, n, pset_len, pset_name, ierr);
  return ;
}


/******************************************************
***      MPI_Session_get_num_psets wrapper function 
******************************************************/
int MPI_Session_get_num_psets(MPI_Session session, MPI_Info info, int* npset_names)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_get_num_psets()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_get_num_psets(session, info, npset_names);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_get_num_psets wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_GET_NUM_PSETS(MPI_Fint session, MPI_Fint info, int* npset_names, MPI_Fint * ierr)
{

  *ierr = MPI_Session_get_num_psets(/* MPI_HANDLE_TYPES */ session, /* MPI_HANDLE_TYPES */ info,
    npset_names);
  return ;
}

/******************************************************
***      MPI_Session_get_num_psets wrapper function (lowercase)
******************************************************/
void mpi_session_get_num_psets(MPI_Fint session, MPI_Fint info, int* npset_names, MPI_Fint * ierr)
{
  MPI_SESSION_GET_NUM_PSETS(session, info, npset_names, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_num_psets wrapper function (lowercase_)
******************************************************/
void mpi_session_get_num_psets_(MPI_Fint session, MPI_Fint info, int* npset_names, MPI_Fint * ierr)
{
  MPI_SESSION_GET_NUM_PSETS(session, info, npset_names, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_num_psets wrapper function (lowercase__)
******************************************************/
void mpi_session_get_num_psets__(MPI_Fint session, MPI_Fint info, int* npset_names, MPI_Fint * ierr)
{
  MPI_SESSION_GET_NUM_PSETS(session, info, npset_names, ierr);
  return ;
}


/******************************************************
***      MPI_Session_get_pset_info wrapper function 
******************************************************/
int MPI_Session_get_pset_info(MPI_Session session, TAU_MPICH3_CONST char* pset_name, MPI_Info* info)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_get_pset_info()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_get_pset_info(session, pset_name, info);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_get_pset_info wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_GET_PSET_INFO(MPI_Fint session, char* pset_name, MPI_Info* info, MPI_Fint * ierr)
{

  *ierr = MPI_Session_get_pset_info(/* MPI_HANDLE_TYPES */ session, pset_name, info);
  return ;
}

/******************************************************
***      MPI_Session_get_pset_info wrapper function (lowercase)
******************************************************/
void mpi_session_get_pset_info(MPI_Fint session, char* pset_name, MPI_Info* info, MPI_Fint * ierr)
{
  MPI_SESSION_GET_PSET_INFO(session, pset_name, info, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_pset_info wrapper function (lowercase_)
******************************************************/
void mpi_session_get_pset_info_(MPI_Fint session, char* pset_name, MPI_Info* info, MPI_Fint * ierr)
{
  MPI_SESSION_GET_PSET_INFO(session, pset_name, info, ierr);
  return ;
}

/******************************************************
***      MPI_Session_get_pset_info wrapper function (lowercase__)
******************************************************/
void mpi_session_get_pset_info__(MPI_Fint session, char* pset_name, MPI_Info* info, MPI_Fint * ierr)
{
  MPI_SESSION_GET_PSET_INFO(session, pset_name, info, ierr);
  return ;
}


/******************************************************
***      MPI_Session_init wrapper function 
******************************************************/
int MPI_Session_init(MPI_Info info, MPI_Errhandler errhandler, MPI_Session* session)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_init()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_init(info, errhandler, session);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_init wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_INIT(MPI_Fint info, MPI_Fint errhandler, MPI_Session* session, MPI_Fint * ierr)
{

  *ierr = MPI_Session_init(/* MPI_HANDLE_TYPES */ info, /* MPI_HANDLE_TYPES */ errhandler, session);
  return ;
}

/******************************************************
***      MPI_Session_init wrapper function (lowercase)
******************************************************/
void mpi_session_init(MPI_Fint info, MPI_Fint errhandler, MPI_Session* session, MPI_Fint * ierr)
{
  MPI_SESSION_INIT(info, errhandler, session, ierr);
  return ;
}

/******************************************************
***      MPI_Session_init wrapper function (lowercase_)
******************************************************/
void mpi_session_init_(MPI_Fint info, MPI_Fint errhandler, MPI_Session* session, MPI_Fint * ierr)
{
  MPI_SESSION_INIT(info, errhandler, session, ierr);
  return ;
}

/******************************************************
***      MPI_Session_init wrapper function (lowercase__)
******************************************************/
void mpi_session_init__(MPI_Fint info, MPI_Fint errhandler, MPI_Session* session, MPI_Fint * ierr)
{
  MPI_SESSION_INIT(info, errhandler, session, ierr);
  return ;
}


/******************************************************
***      MPI_Session_set_errhandler wrapper function 
******************************************************/
int MPI_Session_set_errhandler(MPI_Session session, MPI_Errhandler errhandler)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Session_set_errhandler()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Session_set_errhandler(session, errhandler);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Session_set_errhandler wrapper function (uppercase Fortran)
******************************************************/
void MPI_SESSION_SET_ERRHANDLER(MPI_Fint session, MPI_Fint errhandler, MPI_Fint * ierr)
{

  *ierr = MPI_Session_set_errhandler(/* MPI_HANDLE_TYPES */ session,
    /* MPI_HANDLE_TYPES */ errhandler);
  return ;
}

/******************************************************
***      MPI_Session_set_errhandler wrapper function (lowercase)
******************************************************/
void mpi_session_set_errhandler(MPI_Fint session, MPI_Fint errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_SET_ERRHANDLER(session, errhandler, ierr);
  return ;
}

/******************************************************
***      MPI_Session_set_errhandler wrapper function (lowercase_)
******************************************************/
void mpi_session_set_errhandler_(MPI_Fint session, MPI_Fint errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_SET_ERRHANDLER(session, errhandler, ierr);
  return ;
}

/******************************************************
***      MPI_Session_set_errhandler wrapper function (lowercase__)
******************************************************/
void mpi_session_set_errhandler__(MPI_Fint session, MPI_Fint errhandler, MPI_Fint * ierr)
{
  MPI_SESSION_SET_ERRHANDLER(session, errhandler, ierr);
  return ;
}


/******************************************************
***      MPI_Ssend_c wrapper function 
******************************************************/
int MPI_Ssend_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ssend_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ssend_c(buf, count, datatype, dest, tag, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ssend_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SSEND_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{

  *ierr = MPI_Ssend_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Ssend_c wrapper function (lowercase)
******************************************************/
void mpi_ssend_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Ssend_c wrapper function (lowercase_)
******************************************************/
void mpi_ssend_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Ssend_c wrapper function (lowercase__)
******************************************************/
void mpi_ssend_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Fint * ierr)
{
  MPI_SSEND_C(buf, count, datatype, dest, tag, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Ssend_init_c wrapper function 
******************************************************/
int MPI_Ssend_init_c(TAU_MPICH3_CONST void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
MPI_Request* request)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Ssend_init_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Ssend_init_c(buf, count, datatype, dest, tag, comm, request);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Ssend_init_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_SSEND_INIT_C(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{

  *ierr = MPI_Ssend_init_c(buf, count, /* MPI_HANDLE_TYPES */ datatype, /* MPI_HANDLE_TYPES */ dest,
    /* MPI_HANDLE_TYPES */ tag, /* MPI_HANDLE_TYPES */ comm, request);
  return ;
}

/******************************************************
***      MPI_Ssend_init_c wrapper function (lowercase)
******************************************************/
void mpi_ssend_init_c(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ssend_init_c wrapper function (lowercase_)
******************************************************/
void mpi_ssend_init_c_(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}

/******************************************************
***      MPI_Ssend_init_c wrapper function (lowercase__)
******************************************************/
void mpi_ssend_init_c__(MPI_Aint buf, MPI_Count count, MPI_Fint datatype, MPI_Fint dest, MPI_Fint tag, MPI_Fint comm,
MPI_Request* request, MPI_Fint * ierr)
{
  MPI_SSEND_INIT_C(buf, count, datatype, dest, tag, comm, request, ierr);
  return ;
}


/******************************************************
***      MPI_Status_set_elements_c wrapper function 
******************************************************/
int MPI_Status_set_elements_c(MPI_Status* status, MPI_Datatype datatype, MPI_Count count)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Status_set_elements_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Status_set_elements_c(status, datatype, count);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Status_set_elements_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_STATUS_SET_ELEMENTS_C(MPI_Status* status, MPI_Fint datatype, MPI_Count count, MPI_Fint * ierr)
{

  *ierr = MPI_Status_set_elements_c(status, /* MPI_HANDLE_TYPES */ datatype, count);
  return ;
}

/******************************************************
***      MPI_Status_set_elements_c wrapper function (lowercase)
******************************************************/
void mpi_status_set_elements_c(MPI_Status* status, MPI_Fint datatype, MPI_Count count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS_C(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Status_set_elements_c wrapper function (lowercase_)
******************************************************/
void mpi_status_set_elements_c_(MPI_Status* status, MPI_Fint datatype, MPI_Count count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS_C(status, datatype, count, ierr);
  return ;
}

/******************************************************
***      MPI_Status_set_elements_c wrapper function (lowercase__)
******************************************************/
void mpi_status_set_elements_c__(MPI_Status* status, MPI_Fint datatype, MPI_Count count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS_C(status, datatype, count, ierr);
  return ;
}


/******************************************************
***      MPI_Type_contiguous_c wrapper function 
******************************************************/
int MPI_Type_contiguous_c(MPI_Count count, MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_contiguous_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_contiguous_c(count, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_contiguous_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CONTIGUOUS_C(MPI_Count count, MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{

  *ierr = MPI_Type_contiguous_c(count, /* MPI_HANDLE_TYPES */ oldtype, newtype);
  return ;
}

/******************************************************
***      MPI_Type_contiguous_c wrapper function (lowercase)
******************************************************/
void mpi_type_contiguous_c(MPI_Count count, MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CONTIGUOUS_C(count, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_contiguous_c wrapper function (lowercase_)
******************************************************/
void mpi_type_contiguous_c_(MPI_Count count, MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CONTIGUOUS_C(count, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_contiguous_c wrapper function (lowercase__)
******************************************************/
void mpi_type_contiguous_c__(MPI_Count count, MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CONTIGUOUS_C(count, oldtype, newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Type_create_darray_c wrapper function 
******************************************************/
int MPI_Type_create_darray_c(int size, int rank, int ndims, TAU_MPICH3_CONST MPI_Count array_of_gsizes[], TAU_MPICH3_CONST int array_of_distribs[],
TAU_MPICH3_CONST int array_of_dargs[], TAU_MPICH3_CONST int array_of_psizes[], int order, MPI_Datatype oldtype,
MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_darray_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_create_darray_c(size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs,
    array_of_psizes, order, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_darray_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_DARRAY_C(MPI_Fint size, MPI_Fint rank, MPI_Fint ndims, MPI_Count array_of_gsizes[],
int array_of_distribs[], int array_of_dargs[], int array_of_psizes[],
MPI_Fint order, MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{

  *ierr = MPI_Type_create_darray_c(/* MPI_HANDLE_TYPES */ size, /* MPI_HANDLE_TYPES */ rank,
    /* MPI_HANDLE_TYPES */ ndims, array_of_gsizes, array_of_distribs, array_of_dargs,
    array_of_psizes, /* MPI_HANDLE_TYPES */ order, /* MPI_HANDLE_TYPES */ oldtype, newtype);
  return ;
}

/******************************************************
***      MPI_Type_create_darray_c wrapper function (lowercase)
******************************************************/
void mpi_type_create_darray_c(MPI_Fint size, MPI_Fint rank, MPI_Fint ndims, MPI_Count array_of_gsizes[],
int array_of_distribs[], int array_of_dargs[], int array_of_psizes[],
MPI_Fint order, MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_DARRAY_C(size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs,
    array_of_psizes, order, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_darray_c wrapper function (lowercase_)
******************************************************/
void mpi_type_create_darray_c_(MPI_Fint size, MPI_Fint rank, MPI_Fint ndims, MPI_Count array_of_gsizes[],
int array_of_distribs[], int array_of_dargs[], int array_of_psizes[],
MPI_Fint order, MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_DARRAY_C(size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs,
    array_of_psizes, order, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_darray_c wrapper function (lowercase__)
******************************************************/
void mpi_type_create_darray_c__(MPI_Fint size, MPI_Fint rank, MPI_Fint ndims, MPI_Count array_of_gsizes[],
int array_of_distribs[], int array_of_dargs[], int array_of_psizes[],
MPI_Fint order, MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_DARRAY_C(size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs,
    array_of_psizes, order, oldtype, newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Type_create_hindexed_c wrapper function 
******************************************************/
int MPI_Type_create_hindexed_c(MPI_Count count, TAU_MPICH3_CONST MPI_Count array_of_blocklengths[], TAU_MPICH3_CONST MPI_Count array_of_displacements[],
MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hindexed_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_create_hindexed_c(count, array_of_blocklengths, array_of_displacements, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_hindexed_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_HINDEXED_C(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{

  *ierr = MPI_Type_create_hindexed_c(count, array_of_blocklengths, array_of_displacements,
    /* MPI_HANDLE_TYPES */ oldtype, newtype);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_c wrapper function (lowercase)
******************************************************/
void mpi_type_create_hindexed_c(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_C(count, array_of_blocklengths, array_of_displacements, oldtype, newtype,
    ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_c wrapper function (lowercase_)
******************************************************/
void mpi_type_create_hindexed_c_(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_C(count, array_of_blocklengths, array_of_displacements, oldtype, newtype,
    ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_c wrapper function (lowercase__)
******************************************************/
void mpi_type_create_hindexed_c__(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_C(count, array_of_blocklengths, array_of_displacements, oldtype, newtype,
    ierr);
  return ;
}


/******************************************************
***      MPI_Type_create_indexed_block_c wrapper function 
******************************************************/
int MPI_Type_create_indexed_block_c(MPI_Count count, MPI_Count blocklength, TAU_MPICH3_CONST MPI_Count array_of_displacements[],
MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_indexed_block_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_create_indexed_block_c(count, blocklength, array_of_displacements, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_indexed_block_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_INDEXED_BLOCK_C(MPI_Count count, MPI_Count blocklength, MPI_Count array_of_displacements[], MPI_Fint oldtype,
MPI_Datatype* newtype, MPI_Fint * ierr)
{

  *ierr = MPI_Type_create_indexed_block_c(count, blocklength, array_of_displacements,
    /* MPI_HANDLE_TYPES */ oldtype, newtype);
  return ;
}

/******************************************************
***      MPI_Type_create_indexed_block_c wrapper function (lowercase)
******************************************************/
void mpi_type_create_indexed_block_c(MPI_Count count, MPI_Count blocklength, MPI_Count array_of_displacements[], MPI_Fint oldtype,
MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_INDEXED_BLOCK_C(count, blocklength, array_of_displacements, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_indexed_block_c wrapper function (lowercase_)
******************************************************/
void mpi_type_create_indexed_block_c_(MPI_Count count, MPI_Count blocklength, MPI_Count array_of_displacements[], MPI_Fint oldtype,
MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_INDEXED_BLOCK_C(count, blocklength, array_of_displacements, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_indexed_block_c wrapper function (lowercase__)
******************************************************/
void mpi_type_create_indexed_block_c__(MPI_Count count, MPI_Count blocklength, MPI_Count array_of_displacements[], MPI_Fint oldtype,
MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_INDEXED_BLOCK_C(count, blocklength, array_of_displacements, oldtype, newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Type_create_hindexed_block_c wrapper function 
******************************************************/
int MPI_Type_create_hindexed_block_c(MPI_Count count, MPI_Count blocklength, TAU_MPICH3_CONST MPI_Count array_of_displacements[],
MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hindexed_block_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_create_hindexed_block_c(count, blocklength, array_of_displacements, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_hindexed_block_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_HINDEXED_BLOCK_C(MPI_Count count, MPI_Count blocklength, MPI_Count array_of_displacements[], MPI_Fint oldtype,
MPI_Datatype* newtype, MPI_Fint * ierr)
{

  *ierr = MPI_Type_create_hindexed_block_c(count, blocklength, array_of_displacements,
    /* MPI_HANDLE_TYPES */ oldtype, newtype);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_block_c wrapper function (lowercase)
******************************************************/
void mpi_type_create_hindexed_block_c(MPI_Count count, MPI_Count blocklength, MPI_Count array_of_displacements[], MPI_Fint oldtype,
MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_BLOCK_C(count, blocklength, array_of_displacements, oldtype, newtype,
    ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_block_c wrapper function (lowercase_)
******************************************************/
void mpi_type_create_hindexed_block_c_(MPI_Count count, MPI_Count blocklength, MPI_Count array_of_displacements[], MPI_Fint oldtype,
MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_BLOCK_C(count, blocklength, array_of_displacements, oldtype, newtype,
    ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hindexed_block_c wrapper function (lowercase__)
******************************************************/
void mpi_type_create_hindexed_block_c__(MPI_Count count, MPI_Count blocklength, MPI_Count array_of_displacements[], MPI_Fint oldtype,
MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED_BLOCK_C(count, blocklength, array_of_displacements, oldtype, newtype,
    ierr);
  return ;
}


/******************************************************
***      MPI_Type_create_hvector_c wrapper function 
******************************************************/
int MPI_Type_create_hvector_c(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Datatype oldtype,
MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hvector_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_create_hvector_c(count, blocklength, stride, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_hvector_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_HVECTOR_C(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{

  *ierr = MPI_Type_create_hvector_c(count, blocklength, stride, /* MPI_HANDLE_TYPES */ oldtype,
    newtype);
  return ;
}

/******************************************************
***      MPI_Type_create_hvector_c wrapper function (lowercase)
******************************************************/
void mpi_type_create_hvector_c(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HVECTOR_C(count, blocklength, stride, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hvector_c wrapper function (lowercase_)
******************************************************/
void mpi_type_create_hvector_c_(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HVECTOR_C(count, blocklength, stride, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_hvector_c wrapper function (lowercase__)
******************************************************/
void mpi_type_create_hvector_c__(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HVECTOR_C(count, blocklength, stride, oldtype, newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Type_create_resized_c wrapper function 
******************************************************/
int MPI_Type_create_resized_c(MPI_Datatype oldtype, MPI_Count lb, MPI_Count extent, MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_resized_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_create_resized_c(oldtype, lb, extent, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_resized_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_RESIZED_C(MPI_Fint oldtype, MPI_Count lb, MPI_Count extent, MPI_Datatype* newtype, MPI_Fint * ierr)
{

  *ierr = MPI_Type_create_resized_c(/* MPI_HANDLE_TYPES */ oldtype, lb, extent, newtype);
  return ;
}

/******************************************************
***      MPI_Type_create_resized_c wrapper function (lowercase)
******************************************************/
void mpi_type_create_resized_c(MPI_Fint oldtype, MPI_Count lb, MPI_Count extent, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_RESIZED_C(oldtype, lb, extent, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_resized_c wrapper function (lowercase_)
******************************************************/
void mpi_type_create_resized_c_(MPI_Fint oldtype, MPI_Count lb, MPI_Count extent, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_RESIZED_C(oldtype, lb, extent, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_resized_c wrapper function (lowercase__)
******************************************************/
void mpi_type_create_resized_c__(MPI_Fint oldtype, MPI_Count lb, MPI_Count extent, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_RESIZED_C(oldtype, lb, extent, newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Type_create_struct_c wrapper function 
******************************************************/
int MPI_Type_create_struct_c(MPI_Count count, TAU_MPICH3_CONST MPI_Count array_of_blocklengths[], TAU_MPICH3_CONST MPI_Count array_of_displacements[],
TAU_MPICH3_CONST MPI_Datatype array_of_types[], MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_struct_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_create_struct_c(count, array_of_blocklengths, array_of_displacements, array_of_types,
    newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_struct_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_STRUCT_C(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Datatype array_of_types[], MPI_Datatype* newtype, MPI_Fint * ierr)
{

  *ierr = MPI_Type_create_struct_c(count, array_of_blocklengths, array_of_displacements,
    array_of_types, newtype);
  return ;
}

/******************************************************
***      MPI_Type_create_struct_c wrapper function (lowercase)
******************************************************/
void mpi_type_create_struct_c(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Datatype array_of_types[], MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_STRUCT_C(count, array_of_blocklengths, array_of_displacements, array_of_types,
    newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_struct_c wrapper function (lowercase_)
******************************************************/
void mpi_type_create_struct_c_(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Datatype array_of_types[], MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_STRUCT_C(count, array_of_blocklengths, array_of_displacements, array_of_types,
    newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_struct_c wrapper function (lowercase__)
******************************************************/
void mpi_type_create_struct_c__(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Datatype array_of_types[], MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_STRUCT_C(count, array_of_blocklengths, array_of_displacements, array_of_types,
    newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Type_create_subarray_c wrapper function 
******************************************************/
int MPI_Type_create_subarray_c(int ndims, TAU_MPICH3_CONST MPI_Count array_of_sizes[], TAU_MPICH3_CONST MPI_Count array_of_subsizes[],
TAU_MPICH3_CONST MPI_Count array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_create_subarray_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_create_subarray_c(ndims, array_of_sizes, array_of_subsizes, array_of_starts, order,
    oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_create_subarray_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_CREATE_SUBARRAY_C(MPI_Fint ndims, MPI_Count array_of_sizes[], MPI_Count array_of_subsizes[],
MPI_Count array_of_starts[], MPI_Fint order, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{

  *ierr = MPI_Type_create_subarray_c(/* MPI_HANDLE_TYPES */ ndims, array_of_sizes,
    array_of_subsizes, array_of_starts, /* MPI_HANDLE_TYPES */ order,
    /* MPI_HANDLE_TYPES */ oldtype, newtype);
  return ;
}

/******************************************************
***      MPI_Type_create_subarray_c wrapper function (lowercase)
******************************************************/
void mpi_type_create_subarray_c(MPI_Fint ndims, MPI_Count array_of_sizes[], MPI_Count array_of_subsizes[],
MPI_Count array_of_starts[], MPI_Fint order, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_SUBARRAY_C(ndims, array_of_sizes, array_of_subsizes, array_of_starts, order,
    oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_subarray_c wrapper function (lowercase_)
******************************************************/
void mpi_type_create_subarray_c_(MPI_Fint ndims, MPI_Count array_of_sizes[], MPI_Count array_of_subsizes[],
MPI_Count array_of_starts[], MPI_Fint order, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_SUBARRAY_C(ndims, array_of_sizes, array_of_subsizes, array_of_starts, order,
    oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_create_subarray_c wrapper function (lowercase__)
******************************************************/
void mpi_type_create_subarray_c__(MPI_Fint ndims, MPI_Count array_of_sizes[], MPI_Count array_of_subsizes[],
MPI_Count array_of_starts[], MPI_Fint order, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_SUBARRAY_C(ndims, array_of_sizes, array_of_subsizes, array_of_starts, order,
    oldtype, newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Type_get_contents_c wrapper function 
******************************************************/
int MPI_Type_get_contents_c(MPI_Datatype datatype, MPI_Count max_integers, MPI_Count max_addresses, MPI_Count max_large_counts,
MPI_Count max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[],
MPI_Count array_of_large_counts[], MPI_Datatype array_of_datatypes[])
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_get_contents_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_get_contents_c(datatype, max_integers, max_addresses, max_large_counts, max_datatypes,
    array_of_integers, array_of_addresses, array_of_large_counts, array_of_datatypes);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_get_contents_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_GET_CONTENTS_C(MPI_Fint datatype, MPI_Count max_integers, MPI_Count max_addresses, MPI_Count max_large_counts,
MPI_Count max_datatypes, MPI_Fint array_of_integers[], MPI_Aint array_of_addresses[],
MPI_Count array_of_large_counts[], MPI_Fint array_of_datatypes[], MPI_Fint * ierr)
{

  *ierr = MPI_Type_get_contents_c(/* MPI_HANDLE_TYPES */ datatype, max_integers, max_addresses,
    max_large_counts, max_datatypes, /* MPI_HANDLE_TYPES */ array_of_integers, array_of_addresses,
    array_of_large_counts, /* MPI_HANDLE_TYPES */ array_of_datatypes);
  return ;
}

/******************************************************
***      MPI_Type_get_contents_c wrapper function (lowercase)
******************************************************/
void mpi_type_get_contents_c(MPI_Fint datatype, MPI_Count max_integers, MPI_Count max_addresses, MPI_Count max_large_counts,
MPI_Count max_datatypes, MPI_Fint array_of_integers[], MPI_Aint array_of_addresses[],
MPI_Count array_of_large_counts[], MPI_Fint array_of_datatypes[], MPI_Fint * ierr)
{
  MPI_TYPE_GET_CONTENTS_C(datatype, max_integers, max_addresses, max_large_counts, max_datatypes,
    array_of_integers, array_of_addresses, array_of_large_counts, array_of_datatypes, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_contents_c wrapper function (lowercase_)
******************************************************/
void mpi_type_get_contents_c_(MPI_Fint datatype, MPI_Count max_integers, MPI_Count max_addresses, MPI_Count max_large_counts,
MPI_Count max_datatypes, MPI_Fint array_of_integers[], MPI_Aint array_of_addresses[],
MPI_Count array_of_large_counts[], MPI_Fint array_of_datatypes[], MPI_Fint * ierr)
{
  MPI_TYPE_GET_CONTENTS_C(datatype, max_integers, max_addresses, max_large_counts, max_datatypes,
    array_of_integers, array_of_addresses, array_of_large_counts, array_of_datatypes, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_contents_c wrapper function (lowercase__)
******************************************************/
void mpi_type_get_contents_c__(MPI_Fint datatype, MPI_Count max_integers, MPI_Count max_addresses, MPI_Count max_large_counts,
MPI_Count max_datatypes, MPI_Fint array_of_integers[], MPI_Aint array_of_addresses[],
MPI_Count array_of_large_counts[], MPI_Fint array_of_datatypes[], MPI_Fint * ierr)
{
  MPI_TYPE_GET_CONTENTS_C(datatype, max_integers, max_addresses, max_large_counts, max_datatypes,
    array_of_integers, array_of_addresses, array_of_large_counts, array_of_datatypes, ierr);
  return ;
}


/******************************************************
***      MPI_Type_get_envelope_c wrapper function 
******************************************************/
int MPI_Type_get_envelope_c(MPI_Datatype datatype, MPI_Count* num_integers, MPI_Count* num_addresses,
MPI_Count* num_large_counts, MPI_Count* num_datatypes, int* combiner)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_get_envelope_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_get_envelope_c(datatype, num_integers, num_addresses, num_large_counts, num_datatypes,
    combiner);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_get_envelope_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_GET_ENVELOPE_C(MPI_Fint datatype, MPI_Count* num_integers, MPI_Count* num_addresses, MPI_Count* num_large_counts,
MPI_Count* num_datatypes, int* combiner, MPI_Fint * ierr)
{

  *ierr = MPI_Type_get_envelope_c(/* MPI_HANDLE_TYPES */ datatype, num_integers, num_addresses,
    num_large_counts, num_datatypes, combiner);
  return ;
}

/******************************************************
***      MPI_Type_get_envelope_c wrapper function (lowercase)
******************************************************/
void mpi_type_get_envelope_c(MPI_Fint datatype, MPI_Count* num_integers, MPI_Count* num_addresses, MPI_Count* num_large_counts,
MPI_Count* num_datatypes, int* combiner, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ENVELOPE_C(datatype, num_integers, num_addresses, num_large_counts, num_datatypes,
    combiner, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_envelope_c wrapper function (lowercase_)
******************************************************/
void mpi_type_get_envelope_c_(MPI_Fint datatype, MPI_Count* num_integers, MPI_Count* num_addresses, MPI_Count* num_large_counts,
MPI_Count* num_datatypes, int* combiner, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ENVELOPE_C(datatype, num_integers, num_addresses, num_large_counts, num_datatypes,
    combiner, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_envelope_c wrapper function (lowercase__)
******************************************************/
void mpi_type_get_envelope_c__(MPI_Fint datatype, MPI_Count* num_integers, MPI_Count* num_addresses, MPI_Count* num_large_counts,
MPI_Count* num_datatypes, int* combiner, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ENVELOPE_C(datatype, num_integers, num_addresses, num_large_counts, num_datatypes,
    combiner, ierr);
  return ;
}


/******************************************************
***      MPI_Type_get_extent_c wrapper function 
******************************************************/
int MPI_Type_get_extent_c(MPI_Datatype datatype, MPI_Count* lb, MPI_Count* extent)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_get_extent_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_get_extent_c(datatype, lb, extent);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_get_extent_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_GET_EXTENT_C(MPI_Fint datatype, MPI_Count* lb, MPI_Count* extent, MPI_Fint * ierr)
{

  *ierr = MPI_Type_get_extent_c(/* MPI_HANDLE_TYPES */ datatype, lb, extent);
  return ;
}

/******************************************************
***      MPI_Type_get_extent_c wrapper function (lowercase)
******************************************************/
void mpi_type_get_extent_c(MPI_Fint datatype, MPI_Count* lb, MPI_Count* extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT_C(datatype, lb, extent, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_extent_c wrapper function (lowercase_)
******************************************************/
void mpi_type_get_extent_c_(MPI_Fint datatype, MPI_Count* lb, MPI_Count* extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT_C(datatype, lb, extent, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_extent_c wrapper function (lowercase__)
******************************************************/
void mpi_type_get_extent_c__(MPI_Fint datatype, MPI_Count* lb, MPI_Count* extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT_C(datatype, lb, extent, ierr);
  return ;
}


/******************************************************
***      MPI_Type_get_true_extent_c wrapper function 
******************************************************/
int MPI_Type_get_true_extent_c(MPI_Datatype datatype, MPI_Count* true_lb, MPI_Count* true_extent)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_get_true_extent_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_get_true_extent_c(datatype, true_lb, true_extent);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_get_true_extent_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_GET_TRUE_EXTENT_C(MPI_Fint datatype, MPI_Count* true_lb, MPI_Count* true_extent, MPI_Fint * ierr)
{

  *ierr = MPI_Type_get_true_extent_c(/* MPI_HANDLE_TYPES */ datatype, true_lb, true_extent);
  return ;
}

/******************************************************
***      MPI_Type_get_true_extent_c wrapper function (lowercase)
******************************************************/
void mpi_type_get_true_extent_c(MPI_Fint datatype, MPI_Count* true_lb, MPI_Count* true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT_C(datatype, true_lb, true_extent, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_true_extent_c wrapper function (lowercase_)
******************************************************/
void mpi_type_get_true_extent_c_(MPI_Fint datatype, MPI_Count* true_lb, MPI_Count* true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT_C(datatype, true_lb, true_extent, ierr);
  return ;
}

/******************************************************
***      MPI_Type_get_true_extent_c wrapper function (lowercase__)
******************************************************/
void mpi_type_get_true_extent_c__(MPI_Fint datatype, MPI_Count* true_lb, MPI_Count* true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT_C(datatype, true_lb, true_extent, ierr);
  return ;
}


/******************************************************
***      MPI_Type_indexed_c wrapper function 
******************************************************/
int MPI_Type_indexed_c(MPI_Count count, TAU_MPICH3_CONST MPI_Count array_of_blocklengths[], TAU_MPICH3_CONST MPI_Count array_of_displacements[],
MPI_Datatype oldtype, MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_indexed_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_indexed_c(count, array_of_blocklengths, array_of_displacements, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_indexed_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_INDEXED_C(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{

  *ierr = MPI_Type_indexed_c(count, array_of_blocklengths, array_of_displacements,
    /* MPI_HANDLE_TYPES */ oldtype, newtype);
  return ;
}

/******************************************************
***      MPI_Type_indexed_c wrapper function (lowercase)
******************************************************/
void mpi_type_indexed_c(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_INDEXED_C(count, array_of_blocklengths, array_of_displacements, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_indexed_c wrapper function (lowercase_)
******************************************************/
void mpi_type_indexed_c_(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_INDEXED_C(count, array_of_blocklengths, array_of_displacements, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_indexed_c wrapper function (lowercase__)
******************************************************/
void mpi_type_indexed_c__(MPI_Count count, MPI_Count array_of_blocklengths[], MPI_Count array_of_displacements[],
MPI_Fint oldtype, MPI_Datatype* newtype, MPI_Fint * ierr)
{
  MPI_TYPE_INDEXED_C(count, array_of_blocklengths, array_of_displacements, oldtype, newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Type_size_c wrapper function 
******************************************************/
int MPI_Type_size_c(MPI_Datatype datatype, MPI_Count* size)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_size_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_size_c(datatype, size);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_size_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_SIZE_C(MPI_Fint datatype, MPI_Count* size, MPI_Fint * ierr)
{

  *ierr = MPI_Type_size_c(/* MPI_HANDLE_TYPES */ datatype, size);
  return ;
}

/******************************************************
***      MPI_Type_size_c wrapper function (lowercase)
******************************************************/
void mpi_type_size_c(MPI_Fint datatype, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_TYPE_SIZE_C(datatype, size, ierr);
  return ;
}

/******************************************************
***      MPI_Type_size_c wrapper function (lowercase_)
******************************************************/
void mpi_type_size_c_(MPI_Fint datatype, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_TYPE_SIZE_C(datatype, size, ierr);
  return ;
}

/******************************************************
***      MPI_Type_size_c wrapper function (lowercase__)
******************************************************/
void mpi_type_size_c__(MPI_Fint datatype, MPI_Count* size, MPI_Fint * ierr)
{
  MPI_TYPE_SIZE_C(datatype, size, ierr);
  return ;
}


/******************************************************
***      MPI_Type_vector_c wrapper function 
******************************************************/
int MPI_Type_vector_c(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Datatype oldtype,
MPI_Datatype* newtype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Type_vector_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Type_vector_c(count, blocklength, stride, oldtype, newtype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Type_vector_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_TYPE_VECTOR_C(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{

  *ierr = MPI_Type_vector_c(count, blocklength, stride, /* MPI_HANDLE_TYPES */ oldtype, newtype);
  return ;
}

/******************************************************
***      MPI_Type_vector_c wrapper function (lowercase)
******************************************************/
void mpi_type_vector_c(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_VECTOR_C(count, blocklength, stride, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_vector_c wrapper function (lowercase_)
******************************************************/
void mpi_type_vector_c_(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_VECTOR_C(count, blocklength, stride, oldtype, newtype, ierr);
  return ;
}

/******************************************************
***      MPI_Type_vector_c wrapper function (lowercase__)
******************************************************/
void mpi_type_vector_c__(MPI_Count count, MPI_Count blocklength, MPI_Count stride, MPI_Fint oldtype, MPI_Datatype* newtype,
MPI_Fint * ierr)
{
  MPI_TYPE_VECTOR_C(count, blocklength, stride, oldtype, newtype, ierr);
  return ;
}


/******************************************************
***      MPI_Unpack_c wrapper function 
******************************************************/
int MPI_Unpack_c(TAU_MPICH3_CONST void* inbuf, MPI_Count insize, MPI_Count* position, void* outbuf, MPI_Count outcount,
MPI_Datatype datatype, MPI_Comm comm)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Unpack_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Unpack_c(inbuf, insize, position, outbuf, outcount, datatype, comm);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Unpack_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_UNPACK_C(MPI_Aint inbuf, MPI_Count insize, MPI_Count* position, MPI_Aint outbuf, MPI_Count outcount,
MPI_Fint datatype, MPI_Fint comm, MPI_Fint * ierr)
{

  *ierr = MPI_Unpack_c(inbuf, insize, position, outbuf, outcount, /* MPI_HANDLE_TYPES */ datatype,
    /* MPI_HANDLE_TYPES */ comm);
  return ;
}

/******************************************************
***      MPI_Unpack_c wrapper function (lowercase)
******************************************************/
void mpi_unpack_c(MPI_Aint inbuf, MPI_Count insize, MPI_Count* position, MPI_Aint outbuf, MPI_Count outcount,
MPI_Fint datatype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_UNPACK_C(inbuf, insize, position, outbuf, outcount, datatype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Unpack_c wrapper function (lowercase_)
******************************************************/
void mpi_unpack_c_(MPI_Aint inbuf, MPI_Count insize, MPI_Count* position, MPI_Aint outbuf, MPI_Count outcount,
MPI_Fint datatype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_UNPACK_C(inbuf, insize, position, outbuf, outcount, datatype, comm, ierr);
  return ;
}

/******************************************************
***      MPI_Unpack_c wrapper function (lowercase__)
******************************************************/
void mpi_unpack_c__(MPI_Aint inbuf, MPI_Count insize, MPI_Count* position, MPI_Aint outbuf, MPI_Count outcount,
MPI_Fint datatype, MPI_Fint comm, MPI_Fint * ierr)
{
  MPI_UNPACK_C(inbuf, insize, position, outbuf, outcount, datatype, comm, ierr);
  return ;
}


/******************************************************
***      MPI_Unpack_external_c wrapper function 
******************************************************/
int MPI_Unpack_external_c(TAU_MPICH3_CONST char datarep[], TAU_MPICH3_CONST void* inbuf, MPI_Count insize, MPI_Count* position, void* outbuf,
MPI_Count outcount, MPI_Datatype datatype)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Unpack_external_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Unpack_external_c(datarep, inbuf, insize, position, outbuf, outcount, datatype);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Unpack_external_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_UNPACK_EXTERNAL_C(char datarep[], MPI_Aint inbuf, MPI_Count insize, MPI_Count* position, MPI_Aint outbuf,
MPI_Count outcount, MPI_Fint datatype, MPI_Fint * ierr)
{

  *ierr = MPI_Unpack_external_c(datarep, inbuf, insize, position, outbuf, outcount,
    /* MPI_HANDLE_TYPES */ datatype);
  return ;
}

/******************************************************
***      MPI_Unpack_external_c wrapper function (lowercase)
******************************************************/
void mpi_unpack_external_c(char datarep[], MPI_Aint inbuf, MPI_Count insize, MPI_Count* position, MPI_Aint outbuf,
MPI_Count outcount, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_UNPACK_EXTERNAL_C(datarep, inbuf, insize, position, outbuf, outcount, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_Unpack_external_c wrapper function (lowercase_)
******************************************************/
void mpi_unpack_external_c_(char datarep[], MPI_Aint inbuf, MPI_Count insize, MPI_Count* position, MPI_Aint outbuf,
MPI_Count outcount, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_UNPACK_EXTERNAL_C(datarep, inbuf, insize, position, outbuf, outcount, datatype, ierr);
  return ;
}

/******************************************************
***      MPI_Unpack_external_c wrapper function (lowercase__)
******************************************************/
void mpi_unpack_external_c__(char datarep[], MPI_Aint inbuf, MPI_Count insize, MPI_Count* position, MPI_Aint outbuf,
MPI_Count outcount, MPI_Fint datatype, MPI_Fint * ierr)
{
  MPI_UNPACK_EXTERNAL_C(datarep, inbuf, insize, position, outbuf, outcount, datatype, ierr);
  return ;
}


/******************************************************
***      MPI_Win_allocate_c wrapper function 
******************************************************/
int MPI_Win_allocate_c(MPI_Aint size, MPI_Aint disp_unit, MPI_Info info, MPI_Comm comm, void* baseptr, MPI_Win* win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_allocate_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Win_allocate_c(size, disp_unit, info, comm, baseptr, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_allocate_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_ALLOCATE_C(MPI_Aint size, MPI_Aint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Win* win,
MPI_Fint * ierr)
{

  *ierr = MPI_Win_allocate_c(size, disp_unit, /* MPI_HANDLE_TYPES */ info,
    /* MPI_HANDLE_TYPES */ comm, baseptr, win);
  return ;
}

/******************************************************
***      MPI_Win_allocate_c wrapper function (lowercase)
******************************************************/
void mpi_win_allocate_c(MPI_Aint size, MPI_Aint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Win* win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_C(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_allocate_c wrapper function (lowercase_)
******************************************************/
void mpi_win_allocate_c_(MPI_Aint size, MPI_Aint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Win* win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_C(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_allocate_c wrapper function (lowercase__)
******************************************************/
void mpi_win_allocate_c__(MPI_Aint size, MPI_Aint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Win* win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_C(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_allocate_shared_c wrapper function 
******************************************************/
int MPI_Win_allocate_shared_c(MPI_Aint size, MPI_Aint disp_unit, MPI_Info info, MPI_Comm comm, void* baseptr, MPI_Win* win)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_allocate_shared_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Win_allocate_shared_c(size, disp_unit, info, comm, baseptr, win);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_allocate_shared_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_ALLOCATE_SHARED_C(MPI_Aint size, MPI_Aint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Win* win,
MPI_Fint * ierr)
{

  *ierr = MPI_Win_allocate_shared_c(size, disp_unit, /* MPI_HANDLE_TYPES */ info,
    /* MPI_HANDLE_TYPES */ comm, baseptr, win);
  return ;
}

/******************************************************
***      MPI_Win_allocate_shared_c wrapper function (lowercase)
******************************************************/
void mpi_win_allocate_shared_c(MPI_Aint size, MPI_Aint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Win* win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_SHARED_C(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_allocate_shared_c wrapper function (lowercase_)
******************************************************/
void mpi_win_allocate_shared_c_(MPI_Aint size, MPI_Aint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Win* win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_SHARED_C(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}

/******************************************************
***      MPI_Win_allocate_shared_c wrapper function (lowercase__)
******************************************************/
void mpi_win_allocate_shared_c__(MPI_Aint size, MPI_Aint disp_unit, MPI_Fint info, MPI_Fint comm, MPI_Aint baseptr, MPI_Win* win,
MPI_Fint * ierr)
{
  MPI_WIN_ALLOCATE_SHARED_C(size, disp_unit, info, comm, baseptr, win, ierr);
  return ;
}


/******************************************************
***      MPI_Win_shared_query_c wrapper function 
******************************************************/
int MPI_Win_shared_query_c(MPI_Win win, int rank, MPI_Aint* size, MPI_Aint* disp_unit, void* baseptr)
{
  int retvalue;
  TAU_PROFILE_TIMER(t, "MPI_Win_shared_query_c()", "", TAU_MESSAGE);
  TAU_PROFILE_START(t);
  retvalue = PMPI_Win_shared_query_c(win, rank, size, disp_unit, baseptr);
  TAU_PROFILE_STOP(t);
  return retvalue;
}

/******************************************************
***      MPI_Win_shared_query_c wrapper function (uppercase Fortran)
******************************************************/
void MPI_WIN_SHARED_QUERY_C(MPI_Fint win, MPI_Fint rank, MPI_Aint* size, MPI_Aint* disp_unit, MPI_Aint baseptr, MPI_Fint * ierr)
{

  *ierr = MPI_Win_shared_query_c(/* MPI_HANDLE_TYPES */ win, /* MPI_HANDLE_TYPES */ rank, size,
    disp_unit, baseptr);
  return ;
}

/******************************************************
***      MPI_Win_shared_query_c wrapper function (lowercase)
******************************************************/
void mpi_win_shared_query_c(MPI_Fint win, MPI_Fint rank, MPI_Aint* size, MPI_Aint* disp_unit, MPI_Aint baseptr, MPI_Fint * ierr)
{
  MPI_WIN_SHARED_QUERY_C(win, rank, size, disp_unit, baseptr, ierr);
  return ;
}

/******************************************************
***      MPI_Win_shared_query_c wrapper function (lowercase_)
******************************************************/
void mpi_win_shared_query_c_(MPI_Fint win, MPI_Fint rank, MPI_Aint* size, MPI_Aint* disp_unit, MPI_Aint baseptr, MPI_Fint * ierr)
{
  MPI_WIN_SHARED_QUERY_C(win, rank, size, disp_unit, baseptr, ierr);
  return ;
}

/******************************************************
***      MPI_Win_shared_query_c wrapper function (lowercase__)
******************************************************/
void mpi_win_shared_query_c__(MPI_Fint win, MPI_Fint rank, MPI_Aint* size, MPI_Aint* disp_unit, MPI_Aint baseptr, MPI_Fint * ierr)
{
  MPI_WIN_SHARED_QUERY_C(win, rank, size, disp_unit, baseptr, ierr);
  return ;
}


//https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Iscatterv.3.html
//https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf
//https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node606.htm
//MPI_SESSION_
//MPI_SESSION_{. . . }_ERRHANDLER
//MPI_SESSION_GET_

//https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf