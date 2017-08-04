/****************************************************************************
**            TAU Portable Profiling Package               **
**            http://www.cs.uoregon.edu/research/tau               **
*****************************************************************************
**    Copyright 1997                                    **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**    File         : Profiler.h                      **
**    Description     : TAU Profiling Package                  **
**    Author        : Sameer Shende                      **
**    Contact        : tau-bugs@cs.uoregon.edu                     **
**    Documentation    : See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/
#ifndef TAUCOLLECTIVES_H
#define TAUCOLLECTIVES_H

#if (defined(TAU_WINDOWS))
#pragma warning( disable : 4786 )
#define TAUDECL __cdecl
#else
#define TAUDECL
#endif /* TAU_WINDOWS */

//#ifdef TAU_INCLUDE_MPI_H_HEADER
#ifdef TAU_MPI
#include <mpi.h>
#endif 
//#endif /* TAU_INCLUDE_MPI_H_HEADER */

#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#ifdef TAU_OTF2
#include <otf2/otf2.h>
#else
#warning OTF2 NOT ENABLED
typedef uint8_t OTF2_Type;
#endif

#include <Profile/Profiler.h>
#include <Profile/PapiLayer.h>
//#include <Profile/TulipThreadLayer.h>
//#include <Profile/JNIThreadLayer.h>
//#include <Profile/JVMTIThreadLayer.h>
//#include <Profile/SprocLayer.h>
//#include <Profile/PapiThreadLayer.h>
//#include <Profile/RtsLayer.h>
//#include <Profile/FunctionInfo.h>
//#include <Profile/UserEvent.h>
//#include <Profile/WindowsThreadLayer.h>
//#include <Profile/TauMemory.h>
//#include <Profile/TauScalasca.h>
//#include <Profile/TauCompensate.h>
//#include <Profile/TauHandler.h>
//#include <Profile/TauEnv.h>
//#include <Profile/TauMapping.h>
//#include <Profile/TauSampling.h>

// list of used datatypes
#define TAUCOLLECTIVES_DATATYPES \
    TAUCOLLECTIVES_DATATYPE( BYTE ) \
    TAUCOLLECTIVES_DATATYPE( CHAR ) \
    TAUCOLLECTIVES_DATATYPE( UNSIGNED_CHAR ) \
    TAUCOLLECTIVES_DATATYPE( INT ) \
    TAUCOLLECTIVES_DATATYPE( UNSIGNED ) \
    TAUCOLLECTIVES_DATATYPE( INT32_T ) \
    TAUCOLLECTIVES_DATATYPE( UINT32_T ) \
    TAUCOLLECTIVES_DATATYPE( INT64_T ) \
    TAUCOLLECTIVES_DATATYPE( UINT64_T ) \
    TAUCOLLECTIVES_DATATYPE( DOUBLE )

typedef enum TauCollectives_Datatype
{
    #define TAUCOLLECTIVES_DATATYPE( datatype ) \
    TAUCOLLECTIVES_ ## datatype,
    TAUCOLLECTIVES_DATATYPES
    #undef TAUCOLLECTIVES_DATATYPE
    TAUCOLLECTIVES_NUMBER_OF_DATATYPES
} TauCollectives_Datatype;

#define TAUCOLLECTIVES_OPERATIONS \
    TAUCOLLECTIVES_OPERATION( BAND ) \
    TAUCOLLECTIVES_OPERATION( BOR ) \
    TAUCOLLECTIVES_OPERATION( MIN ) \
    TAUCOLLECTIVES_OPERATION( MAX ) \
    TAUCOLLECTIVES_OPERATION( SUM )

typedef enum TauCollectives_Operation
{
    #define TAUCOLLECTIVES_OPERATION( op ) \
    TAUCOLLECTIVES_ ## op,
    TAUCOLLECTIVES_OPERATIONS
    #undef TAUCOLLECTIVES_OPERATION
    TAUCOLLECTIVES_NUMBER_OF_OPERATIONS
} TauCollectives_Operation;



#ifdef TAU_SHMEM
#include <shmem.h>
struct TauCollectives_Group
{
  int pe_start;
  int log_pe_stride;
  int pe_size;
  int is_group_set;
};
#elif TAU_MPI
struct TauCollectives_Group
{
  MPI_Comm comm;
};
#else
struct TauCollectives_Group
{
  int dummy;
};
#endif


void TauCollectives_Init(void);
void TauCollectives_Finalize( void );
int TauCollectives_get_size(TauCollectives_Group* group);
TauCollectives_Datatype TauCollectives_get_type(OTF2_Type type);
int TauCollectives_Barrier(TauCollectives_Group* group);
int TauCollectives_Bcast(TauCollectives_Group* group, void* buf, int count, TauCollectives_Datatype datatype, int root);
int TauCollectives_Gather(TauCollectives_Group* group, const void* sendbuf, void* recvbuf, int count,
                  TauCollectives_Datatype datatype, int root);
int TauCollectives_Gatherv(TauCollectives_Group* group, const void* sendbuf, int sendcount,
                   void* recvbuf, const int* recvcnts, TauCollectives_Datatype datatype, int root );
int TauCollectives_Allgather(TauCollectives_Group* group,  const void* sendbuf, void* recvbuf, int count, TauCollectives_Datatype datatype);
int TauCollectives_Reduce(TauCollectives_Group* group, const void* sendbuf, void* recvbuf, int count,
                  TauCollectives_Datatype datatype, TauCollectives_Operation operation, int root);
int TauCollectives_Allreduce(TauCollectives_Group* group, const void* sendbuf, void* recvbuf, int count,
                     TauCollectives_Datatype datatype, TauCollectives_Operation operation);
int TauCollectives_Scatter(TauCollectives_Group* group, const void* sendbuf, void* recvbuf, int count,
                   TauCollectives_Datatype datatype, int root);
int TauCollectives_Scatterv(TauCollectives_Group* group, const void* sendbuf, const int* sendcounts,
                    void* recvbuf, int recvcount, TauCollectives_Datatype datatype, int root);
TauCollectives_Group * TauCollectives_Get_World();

#endif /* TAUCOLLECTIVES_H */

