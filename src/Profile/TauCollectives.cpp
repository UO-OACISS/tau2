/****************************************************************************
**            TAU Portable Profiling Package               **
**            http://www.cs.uoregon.edu/research/tau               **
*****************************************************************************
**    Copyright 1997-2017                               **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**    File         : TauCollectives.cpp                  **
**    Description     : TAU Profiling Package Collective commands      **
**    Author          : Samuel Khuvis                   **
**    Contact        : tau-bugs@cs.uoregon.edu                **
**    Documentation    : See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#include <Profile/TauCollectives.h>
//#include <Profile/TauSampling.h>
//#include <Profile/TauMetrics.h>
//#include <Profile/TauSnapshot.h>
//#include <Profile/TauTrace.h>

#ifdef TAU_DOT_H_LESS_HEADERS 
#include <sstream>
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <sstream.h>
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <stdio.h> 
#include <time.h>
#include <stdlib.h>
#include <limits.h>

#include <string>

using namespace std;
//using namespace tau;

#ifdef TAU_SHMEM
static void*   symmetric_buffer_a;
static void*   symmetric_buffer_b;
static int*    transfer_counter;
static int*    transfer_status;
static int*    current_ready_pe;
static long*   barrier_psync;
static long*   bcast_psync;
static long*   collect_psync;
static long*   reduce_psync;
static double* pwork;
static size_t  current_pwork_size;
#endif /* TAU_SHMEM */

#ifdef TAU_MPI
static MPI_Datatype mpi_datatypes[ TAUCOLLECTIVES_NUMBER_OF_DATATYPES ];
#endif /* TAU_MPI */
TauCollectives_Group tau_group_world;

#define CEIL( a, b )       ( ( ( a ) / ( b ) ) + ( ( ( a ) % ( b ) ) > 0 ? 1 : 0 ) )
#define ROUNDUPTO( a, b )  ( CEIL( a, b ) * b )
#define MIN( a, b )        ( a > b ? a : b )

void
TauCollectives_Init(void) {
#ifdef TAU_SHMEM
  tau_group_world.pe_start      = 0;
  tau_group_world.log_pe_stride = 0;
  tau_group_world.pe_size       = shmem_n_pes();
  tau_group_world.is_group_set  = 1;

  /* Allocate memory in symmetric heap */
  symmetric_buffer_a = shmem_malloc ( BUFFER_SIZE );

  symmetric_buffer_b =  shmem_malloc ( BUFFER_SIZE );

  transfer_status = ( int* ) shmem_malloc ( sizeof( int ) );
  *transfer_status = -1;

  current_ready_pe = ( int* ) shmem_malloc ( sizeof( int ) );
  *current_ready_pe = -1;

  transfer_counter = ( int* ) shmem_malloc ( shmem_n_pes()* sizeof( int ) );
  memset( transfer_counter, 0, shmem_n_pes() * sizeof( int ) );

  barrier_psync = shmem_malloc ( _SHMEM_BARRIER_SYNC_SIZE * sizeof( long ) );
  for ( uint32_t i = 0; i < _SHMEM_BARRIER_SYNC_SIZE; i++ )
  {
    barrier_psync[ i ] = _SHMEM_SYNC_VALUE;
  }

  bcast_psync = shmem_malloc ( _SHMEM_BCAST_SYNC_SIZE * sizeof( long ) );
  for ( uint32_t i = 0; i < _SHMEM_BCAST_SYNC_SIZE; i++ )
  {
    bcast_psync[ i ] = _SHMEM_SYNC_VALUE;
  }

  collect_psync = shmem_malloc ( _SHMEM_COLLECT_SYNC_SIZE * sizeof( long ) );
  for ( uint32_t i = 0; i < _SHMEM_COLLECT_SYNC_SIZE; i++ )
  {
    collect_psync[ i ] = _SHMEM_SYNC_VALUE;
  }

  reduce_psync = shmem_malloc ( _SHMEM_REDUCE_SYNC_SIZE * sizeof( long ) );
  for ( uint32_t i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; i++ )
  {
    reduce_psync[ i ] = _SHMEM_SYNC_VALUE;
  }

  current_pwork_size = _SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof( double );
  pwork              =  shmem_malloc ( current_pwork_size );

  shmem_barrier_all ();
#endif /* TAU_SHMEM */

#ifdef TAU_MPI
  PMPI_Comm_dup( MPI_COMM_WORLD, &tau_group_world.comm );
#define TAUCOLLECTIVES_MPI_BYTE          MPI_BYTE
#define TAUCOLLECTIVES_MPI_CHAR          MPI_CHAR
#define TAUCOLLECTIVES_MPI_UNSIGNED_CHAR MPI_UNSIGNED_CHAR
#define TAUCOLLECTIVES_MPI_INT           MPI_INT
#define TAUCOLLECTIVES_MPI_UNSIGNED      MPI_UNSIGNED
#define TAUCOLLECTIVES_MPI_DOUBLE        MPI_DOUBLE
#define TAUCOLLECTIVES_MPI_INT32_T       MPI_INT32_T
#define TAUCOLLECTIVES_MPI_UINT32_T      MPI_UINT32_T
#define TAUCOLLECTIVES_MPI_INT64_T       MPI_INT64_T
#define TAUCOLLECTIVES_MPI_UINT64_T      MPI_UINT64_T

#define TAUCOLLECTIVES_DATATYPE( datatype ) \
      mpi_datatypes[ TAUCOLLECTIVES_ ## datatype ] = TAUCOLLECTIVES_MPI_ ## datatype;
      TAUCOLLECTIVES_DATATYPES
#undef TAUCOLLECTIVES_DATATYPE

#undef TAUCOLLECTIVES_MPI_BYTE
#undef TAUCOLLECTIVES_MPI_CHAR
#undef TAUCOLLECTIVES_MPI_UNSIGNED_CHAR
#undef TAUCOLLECTIVES_MPI_INT
#undef TAUCOLLECTIVES_MPI_UNSIGNED
#undef TAUCOLLECTIVES_MPI_DOUBLE
#endif /* TAU_MPI */
}

void
TauCollectives_Finalize( void )
{
#ifdef TAU_SHMEM
    shmem_free ( symmetric_buffer_a );
    symmetric_buffer_a = NULL;

    shmem_free ( symmetric_buffer_b );
    symmetric_buffer_b = NULL;

    shmem_free ( transfer_status );
    transfer_status = NULL;

    shmem_free ( current_ready_pe );
    current_ready_pe = NULL;

    shmem_free ( transfer_counter );
    transfer_counter = NULL;

    shmem_free ( barrier_psync );
    barrier_psync = NULL;

    shmem_free ( bcast_psync );
    bcast_psync = NULL;

    shmem_free ( collect_psync );
    collect_psync = NULL;

    shmem_free ( reduce_psync );
    reduce_psync = NULL;

    shmem_free ( pwork );
    pwork = NULL;

    shmem_barrier_all ();
#endif /* TAU_SHMEM */
#ifdef TAU_MPI
    PMPI_Comm_free( &tau_group_world.comm );
#endif /* TAU_MPI */
}

TauCollectives_Datatype
TauCollectives_get_type(OTF2_Type type)
{
  switch ( type )
  {
    case OTF2_TYPE_INT8:
      return TAUCOLLECTIVES_CHAR;
    case OTF2_TYPE_UINT8:
      return TAUCOLLECTIVES_UNSIGNED_CHAR;
    case OTF2_TYPE_INT32:
      return TAUCOLLECTIVES_INT32_T;
    case OTF2_TYPE_UINT32:
      return TAUCOLLECTIVES_UINT32_T;
    case OTF2_TYPE_INT64:
      return TAUCOLLECTIVES_INT64_T;
    case OTF2_TYPE_UINT64:
      return TAUCOLLECTIVES_UINT64_T;
    case OTF2_TYPE_DOUBLE:
      return TAUCOLLECTIVES_DOUBLE;

    default:
      fprintf(stderr, "Unhandled OTF2 type: %u", type );
      return TAUCOLLECTIVES_CHAR;
  }
}

#ifdef TAU_MPI
static inline MPI_Datatype
get_mpi_datatype( TauCollectives_Datatype datatype )
{
  return mpi_datatypes[ datatype ];
}

static inline MPI_Op
get_mpi_operation(TauCollectives_Operation op )
{
  switch ( op )
  {
#define TAUCOLLECTIVES_OPERATION( op ) \
  case TAUCOLLECTIVES_ ## op: \
    return MPI_ ## op;
    TAUCOLLECTIVES_OPERATIONS
#undef TAUCOLLECTIVES_OPERATION
    default:
      fprintf(stderr, "Unknown reduction operation: %u", op);
  }

  return MPI_OP_NULL;
}
#endif /* TAU_MPI */

int TauCollectives_get_size(TauCollectives_Group *group) {
  TauInternalFunctionGuard protects_this_function;
  int size = 1;
#ifdef TAU_MPI
  PMPI_Comm_size(group->comm, &size);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  size = group->pe_size;
#endif /* TAU_SHMEM */
  return size;
}

int TauCollectives_Barrier(TauCollectives_Group *group) {
#ifdef TAU_MPI
  return PMPI_Barrier(group->comm);
#elif TAU_SHMEM
  return shmem_barrier(group->pe_start, group->log_pe_stride, group->pe_size, barrier_psync);
#else
  return 0;
#endif
}

int TauCollectives_Bcast(TauCollectives_Group*   group,
                         void*                   buf,
                         int                     count,
                         TauCollectives_Datatype datatype,
                         int                     root)
{
#ifdef TAU_MPI
  return PMPI_Bcast(buf, count, get_mpi_datatype(datatype), root, group->comm);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  int rank         = RtsLayer::myNode();
  int start        = group->pe_start;
  int stride       = group->log_pe_stride;
  int size         = group->pe_size;
  int num_elements = count;

  if ( datatype == TAUCOLLECTIVES_BYTE ||
       datatype == TAUCOLLECTIVES_CHAR ||
       datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
  {
    num_elements = ROUNDUPTO( count, 4 );
  }

  /* Copy buffer to symmetric memory block */
  if ( root == rank )
  {
      memcpy( symmetric_buffer_a, buf, count * sizeof_ipc_datatypes[ datatype ] );
  }

  shmem_barrier( start, stride, size, barrier_psync );

  int num_elements_sent;
  switch ( datatype )
  {
    case TAUCOLLECTIVES_BYTE:
    case TAUCOLLECTIVES_CHAR:
    case TAUCOLLECTIVES_UNSIGNED_CHAR:
      num_elements_sent = CEIL( count, 4 );
      shmem_broadcast32 ( symmetric_buffer_a,
                          symmetric_buffer_a,
                          num_elements_sent,
                          root, start, stride, size,
                          bcast_psync );
      break;

    case TAUCOLLECTIVES_INT:
    case TAUCOLLECTIVES_UNSIGNED:
    case TAUCOLLECTIVES_INT32_T:
    case TAUCOLLECTIVES_UINT32_T:
      shmem_broadcast32( symmetric_buffer_a,
                         symmetric_buffer_a,
                         count,
                         root, start, stride, size,
                         bcast_psync );
      break;

    case TAUCOLLECTIVES_INT64_T:
    case TAUCOLLECTIVES_UINT64_T:
    case TAUCOLLECTIVES_DOUBLE:
      shmem_broadcast64( symmetric_buffer_a,
                         symmetric_buffer_a,
                         count,
                         root, start, stride, size,
                         bcast_psync );
      break;

    default:
      fprintf(stderr, "Bcast: Invalid datatype: %d", datatype );
  }
  shmem_barrier ( start, stride, size, barrier_psync );

  if ( root != rank )
  {
    /* Copy symmetric memory block to buffer */
    memcpy( buf, symmetric_buffer_a, count * sizeof_ipc_datatypes[ datatype ] );
  }

  shmem_barrier ( start, stride, size, barrier_psync );
#endif /* TAU_SHMEM */
  return 0;
}

int TauCollectives_Gather(TauCollectives_Group*   group,
                          const void*             sendbuf,
                          void*                   recvbuf,
                          int                     count,
                          TauCollectives_Datatype datatype,
                          int                     root )
{
#ifdef TAU_MPI
  return PMPI_Gather((void*) sendbuf, count, get_mpi_datatype(datatype),
                     recvbuf, count, get_mpi_datatype(datatype), root, group->comm);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  int rank         = RtsLayer::myNode();
  int start        = group->pe_start;
  int stride       = group->log_pe_stride;
  int size         = group->pe_size;
  int num_elements = count;

  if ( datatype == TAUCOLLECTIVES_BYTE ||
       datatype == TAUCOLLECTIVES_CHAR ||
       datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
  {
      num_elements = ROUNDUPTO( count, 4 );
  }

  /* Copy buffer to symmetric memory block */
  memcpy( symmetric_buffer_a, sendbuf, count * sizeof_ipc_datatypes[ datatype ] );

  shmem_barrier(start, stride, size, barrier_psync);

  /* Broadcast operation */
  int num_elements_sent;
  switch ( datatype )
  {
      case TAUCOLLECTIVES_BYTE:
      case TAUCOLLECTIVES_CHAR:
      case TAUCOLLECTIVES_UNSIGNED_CHAR:
          num_elements_sent = CEIL( count, 4 );
          shmem_fcollect32(symmetric_buffer_b,
                           symmetric_buffer_a,
                           num_elements_sent,
                           start, stride, size,
                           collect_psync );
          break;

      case TAUCOLLECTIVES_INT:
      case TAUCOLLECTIVES_UNSIGNED:
      case TAUCOLLECTIVES_INT32_T:
      case TAUCOLLECTIVES_UINT32_T:
        CALL_SHMEM( shmem_fcollect32 )( symmetric_buffer_b,
                                        symmetric_buffer_a,
                                        count,
                                        start, stride, size,
                                        collect_psync );
        break;

      case TAUCOLLECTIVES_INT64_T:
      case TAUCOLLECTIVES_UINT64_T:
      case TAUCOLLECTIVES_DOUBLE:
        CALL_SHMEM( shmem_fcollect64 )( symmetric_buffer_b,
                                        symmetric_buffer_a,
                                        count,
                                        start, stride, size,
                                        collect_psync );
        break;

        default:
          fprintf(stderr, "Gather: Invalid datatype: %d", datatype );
  }

  shmem_barrier(start, stride, size, barrier_psync);

  /* Copy symmetric memory block to buffer */
  if ( rank == root )
  {
      if ( datatype == TAUCOLLECTIVES_BYTE ||
           datatype == TAUCOLLECTIVES_CHAR ||
           datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
      {
          int recvbuf_index = 0;
          for ( int i = 0; i < size; i++ )
          {
              for ( int j = 0; j < count; j++ )
              {
                  ((char*)recvbuf)[recvbuf_index++] = ((char*)symmetric_buffer_b)[j + (i*count)];
              }
          }
      }
      else
      {
          memcpy(recvbuf, symmetric_buffer_b, size * count * sizeof_ipc_datatypes[datatype]);
      }
  }

  shmem_barrier(start, stride, size, barrier_psync);

#endif /* TAU_SHMEM */
  return 0;
}

int TauCollectives_Gatherv(TauCollectives_Group*   group,
                       const void*                 sendbuf,
                       int                         sendcount,
                       void*                       recvbuf,
                       const int*                  recvcnts,
                       TauCollectives_Datatype     datatype,
                       int                         root )
{
#ifdef TAU_MPI
  int* displs = NULL;
  int  rank   = RtsLayer::myNode();
  if ( root == rank )
  {
    int size = TauCollectives_get_size(group);
    displs = (int *)calloc( size, sizeof( *displs ) );

    int total = 0;
    for(int i = 0; i<size; i++)
    {
      displs[i] = total;
      total    += recvcnts[i];
    }
  }

  int ret = PMPI_Gatherv((void*) sendbuf,
                         sendcount,
                         get_mpi_datatype( datatype ),
                         recvbuf,
                         (int*) recvcnts,
                         displs,
                         get_mpi_datatype( datatype ),
                         root,
                         group->comm);

  free(displs);

  return ret;
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  int rank         = RtsLayer::myNode();
  int start        = group->pe_start;
  int stride       = group->log_pe_stride;
  int size         = group->pe_size;


  int sendcount_extra = 0;
#if HAVE( BROKEN_SHMEM_COLLECT )
  sendcount_extra = 1;
#endif

  /*
   * Sum up 'recvcount's from all processing elements
   * and allocate symmetric memory blocks
   */
  int total_number_of_recv_elements = 0;
  if (datatype == TAUCOLLECTIVES_BYTE ||
      datatype == TAUCOLLECTIVES_CHAR ||
      datatype == TAUCOLLECTIVES_UNSIGNED_CHAR)
  {
    int num_send_elements = ROUNDUPTO( sendcount + sendcount_extra, 4 );

    if ( rank == root )
    {
      for ( int i = 0; i < TauCollectives_get_size(); i++ )
      {
        int num_recv_elements = ROUNDUPTO( recvcnts[ i ] + sendcount_extra, 4 );
        total_number_of_recv_elements += num_recv_elements;
      }
    }
  }
  else
  {
    if ( rank == root )
    {
      for ( int i = 0; i < TauCollectives_get_size(); i++ )
      {
        total_number_of_recv_elements += recvcnts[i] + sendcount_extra;
      }
    }
  }

  /* Copy buffer to symmetric memory block */
  memcpy( symmetric_buffer_a, sendbuf, sendcount * sizeof_ipc_datatypes[ datatype ] );
  memset( ( char* )symmetric_buffer_a + sendcount * sizeof_ipc_datatypes[ datatype ],
          0, sendcount_extra * sizeof_ipc_datatypes[ datatype ] );

  shmem_barrier(start, stride, size, barrier_psync);

  /* Broadcast operation */
  int num_elements_sent;
  switch ( datatype )
  {
    case TAUCOLLECTIVES_BYTE:
    case TAUCOLLECTIVES_CHAR:
    case TAUCOLLECTIVES_UNSIGNED_CHAR:
      num_elements_sent = CEIL( sendcount + sendcount_extra, 4 );
      shmem_collect32(symmetric_buffer_b,
                      symmetric_buffer_a,
                      num_elements_sent,
                      start, stride, size,
                      collect_psync );
      break;

    case TAUCOLLECTIVES_INT:
    case TAUCOLLECTIVES_UNSIGNED:
    case TAUCOLLECTIVES_INT32_T:
    case TAUCOLLECTIVES_UINT32_T:
      shmem_collect32(symmetric_buffer_b,
                      symmetric_buffer_a,
                      sendcount + sendcount_extra,
                      start, stride, size,
                      collect_psync );
      break;

    case TAUCOLLECTIVES_INT64_T:
    case TAUCOLLECTIVES_UINT64_T:
    case TAUCOLLECTIVES_DOUBLE:
      shmem_collect64(symmetric_buffer_b,
                      symmetric_buffer_a,
                      sendcount + sendcount_extra,
                      start, stride, size,
                      collect_psync);
      break;

    default:
      fprintf(stderr, "Gatherv: Invalid datatype: %d", datatype);
  }

  shmem_barrier(start, stride, size, barrier_psync);

  /* Copy symmetric memory block to buffer */
  if ( rank == root )
  {
    if ( datatype == TAUCOLLECTIVES_BYTE ||
         datatype == TAUCOLLECTIVES_CHAR ||
         datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
    {
      int recvbuf_index  = 0;
      int current_offset = 0;
      for ( int i = 0; i < size; i++ )
      {
        for ( int j = 0; j < recvcnts[ i ]; j++ )
        {
          ((char*)recvbuf)[recvbuf_index++] = ((char*)symmetric_buffer_b)[j + current_offset];
        }
        current_offset += ROUNDUPTO( recvcnts[ i ] + sendcount_extra, 4 );
      }
    }
    else
    {
      size_t recvbuf_offset   = 0;
      size_t symmetric_offset = 0;
      for ( int i = 0; i < size; i++ )
      {
        memcpy((char*)recvbuf + recvbuf_offset,
               (char*)symmetric_buffer_b + symmetric_offset,
               recvcnts[i] * sizeof_ipc_datatypes[datatype]);
        recvbuf_offset   += recvcnts[i] * sizeof_ipc_datatypes[datatype];
        symmetric_offset += (recvcnts[i] + sendcount_extra ) * sizeof_ipc_datatypes[datatype];
      }
    }
  }

  shmem_barrier(start, stride, size, barrier_psync);
#endif /* TAU_SHMEM */
  return 0;
}

int TauCollectives_Allgather(TauCollectives_Group*   group,
                         const void*                 sendbuf,
                         void*                       recvbuf,
                         int                         count,
                         TauCollectives_Datatype     datatype)
{
#ifdef TAU_MPI
  return PMPI_Allgather((void*) sendbuf,
                        count,
                        get_mpi_datatype( datatype ),
                        recvbuf,
                        count,
                        get_mpi_datatype( datatype ),
                        group->comm);

#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  if ( count <= 0 )
  {
      return 0;
  }

  int start        = group->pe_start;
  int stride       = group->log_pe_stride;
  int size         = group->pe_size;
  int num_elements = count;

  if ( datatype == TAUCOLLECTIVES_BYTE ||
       datatype == TAUCOLLECTIVES_CHAR ||
       datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
  {
      num_elements = ROUNDUPTO( count, 4 );
  }

  /* Copy buffer to symmetric memory block */
  memcpy(symmetric_buffer_a, sendbuf, count * sizeof_ipc_datatypes[ datatype ]);

  shmem_barrier( start, stride, size, barrier_psync );

  /* Broadcast operation */
  int num_elements_sent;
  switch ( datatype )
  {
      case TAUCOLLECTIVES_BYTE:
      case TAUCOLLECTIVES_CHAR:
      case TAUCOLLECTIVES_UNSIGNED_CHAR:
        num_elements_sent = CEIL( count, 4 );
        shmem_fcollect32(symmetric_buffer_b,
                         symmetric_buffer_a,
                         num_elements_sent,
                         start, stride, size,
                         collect_psync );
        break;

      case TAUCOLLECTIVES_INT:
      case TAUCOLLECTIVES_UNSIGNED:
      case TAUCOLLECTIVES_INT32_T:
      case TAUCOLLECTIVES_UINT32_T:
        shmem_fcollect32(symmetric_buffer_b,
                         symmetric_buffer_a,
                         count,
                         start, stride, size,
                         collect_psync );
        break;

      case TAUCOLLECTIVES_INT64_T:
      case TAUCOLLECTIVES_UINT64_T:
      case TAUCOLLECTIVES_DOUBLE:
        shmem_fcollect64(symmetric_buffer_b,
                         symmetric_buffer_a,
                         count,
                         start, stride, size,
                         collect_psync);
        break;

      default:
        UTILS_BUG( "Allgather: Invalid datatype: %d", datatype );
  }

  shmem_barrier(start, stride, size, barrier_psync);

  /* Copy symmetric memory block to buffer */
  if (datatype == TAUCOLLECTIVES_BYTE ||
      datatype == TAUCOLLECTIVES_CHAR ||
      datatype == TAUCOLLECTIVES_UNSIGNED_CHAR)
  {
    int recvbuf_index = 0;
    for ( int i = 0; i < size; i++ )
    {
      for ( int j = 0; j < count; j++ )
      {
        ((char*)recvbuf)[recvbuf_index++] = ((char*)symmetric_buffer_b)[j + (i * count)];
      }
    }
  }
  else
  {
    memcpy(recvbuf, symmetric_buffer_b, size * count * sizeof_ipc_datatypes[datatype]);
  }

  shmem_barrier(start, stride, size, barrier_psync);
#endif /* TAU_SHMEM */
  return 0;
}

int TauCollectives_Reduce(TauCollectives_Group*    group,
                          const void*              sendbuf,
                          void*                    recvbuf,
                          int                      count,
                          TauCollectives_Datatype  datatype,
                          TauCollectives_Operation operation,
                          int                      root )
{
#ifdef TAU_MPI
  return PMPI_Reduce((void*) sendbuf,
                     recvbuf,
                     count,
                     get_mpi_datatype(datatype),
                     get_mpi_operation(operation),
                     root,
                     group->comm);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  if ( count <= 0 )
  {
    return 0;
  }

  int rank         = RtsLayer::myNode();
  int start        = group->pe_start;
  int stride       = group->log_pe_stride;
  int size         = group->pe_size;
  int num_elements = count;

  if ( datatype == TAUCOLLECTIVES_BYTE ||
       datatype == TAUCOLLECTIVES_CHAR ||
       datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
  {
    num_elements = ROUNDUPTO( count, 2 );
  }

  /* Copy buffer to symmetric memory block */
  memcpy(symmetric_buffer_a, sendbuf, count * sizeof_ipc_datatypes[datatype]);

  shmem_barrier(start, stride, size, barrier_psync);

  /* Reduction operation */
  if ( datatype == TAUCOLLECTIVES_BYTE ||
       datatype == TAUCOLLECTIVES_CHAR ||
       datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
  {
    int nreduce = CEIL( count, 2 );
    switch ( operation )
    {
      case TAUCOLLECTIVES_BAND:
        shmem_short_and_to_all(symmetric_buffer_b,
                               symmetric_buffer_a,
                               nreduce,
                               start, stride, size,
                               get_pwork( sizeof( short ), nreduce ),
                               reduce_psync );
        break;
      case TAUCOLLECTIVES_BOR:
        shmem_short_or_to_all(symmetric_buffer_b,
                              symmetric_buffer_a,
                              nreduce,
                              start, stride, size,
                              get_pwork( sizeof( short ), nreduce ),
                              reduce_psync );
        break;
      case TAUCOLLECTIVES_MIN:
        shmem_short_min_to_all(symmetric_buffer_b,
                               symmetric_buffer_a,
                               nreduce,
                               start, stride, size,
                               get_pwork( sizeof( short ), nreduce ),
                               reduce_psync );
        break;
      case TAUCOLLECTIVES_MAX:
        shmem_short_max_to_all(symmetric_buffer_b,
                               symmetric_buffer_a,
                               nreduce,
                               start, stride, size,
                               get_pwork( sizeof( short ), nreduce ),
                               reduce_psync );
        break;
      case TAUCOLLECTIVES_SUM:
        shmem_short_sum_to_all(symmetric_buffer_b,
                               symmetric_buffer_a,
                               nreduce,
                               start, stride, size,
                               get_pwork( sizeof( short ), nreduce ),
                               reduce_psync );
        break;
      default:
        fprintf(stderr, "Reduce: Invalid operation: %d", operation );
    }
  }
  else if ( datatype == TAUCOLLECTIVES_INT ||
            datatype == TAUCOLLECTIVES_UNSIGNED ||
            datatype == TAUCOLLECTIVES_INT32_T ||
            datatype == TAUCOLLECTIVES_UINT32_T )
  {
    switch ( operation )
    {
        case TAUCOLLECTIVES_BAND:
            shmem_int_and_to_all(symmetric_buffer_b,
                                 symmetric_buffer_a,
                                 count,
                                 start, stride, size,
                                 get_pwork(sizeof(int), count),
                                 reduce_psync);
            break;
        case TAUCOLLECTIVES_BOR:
            shmem_int_or_to_all(symmetric_buffer_b,
                                symmetric_buffer_a,
                                count,
                                start, stride, size,
                                get_pwork(sizeof(int), count),
                                reduce_psync);
            break;
        case TAUCOLLECTIVES_MIN:
            shmem_int_min_to_all(symmetric_buffer_b,
                                 symmetric_buffer_a,
                                 count,
                                 start, stride, size,
                                 get_pwork(sizeof(int), count),
                                 reduce_psync);
            break;
        case TAUCOLLECTIVES_MAX:
            shmem_int_max_to_all(symmetric_buffer_b,
                                 symmetric_buffer_a,
                                 count,
                                 start, stride, size,
                                 get_pwork(sizeof(int), count),
                                 reduce_psync);
            break;
        case TAUCOLLECTIVES_SUM:
            shmem_int_sum_to_all(symmetric_buffer_b,
                                 symmetric_buffer_a,
                                 count,
                                 start, stride, size,
                                 get_pwork(sizeof(int), count),
                                 reduce_psync);
            break;
        default:
            fprintf(stderr, "Reduce: Invalid operation: %d", operation );
    }
  }
  else if ( datatype == TAUCOLLECTIVES_INT64_T ||
            datatype == TAUCOLLECTIVES_UINT64_T ||
            datatype == TAUCOLLECTIVES_DOUBLE )
  {
    switch ( operation )
    {
        case TAUCOLLECTIVES_BAND:
            shmem_longlong_and_to_all(symmetric_buffer_b,
                                      symmetric_buffer_a,
                                      count,
                                      start, stride, size,
                                      (long long*)pwork,
                                      reduce_psync);
            break;
        case TAUCOLLECTIVES_BOR:
            shmem_longlong_or_to_all(symmetric_buffer_b,
                                     symmetric_buffer_a,
                                     count,
                                     start, stride, size,
                                     (long long*)pwork,
                                     reduce_psync);
            break;
        case TAUCOLLECTIVES_MIN:
            shmem_longlong_min_to_all(symmetric_buffer_b,
                                      symmetric_buffer_a,
                                      count,
                                      start, stride, size,
                                      (long long*)pwork,
                                      reduce_psync);
            break;
        case TAUCOLLECTIVES_MAX:
            shmem_longlong_max_to_all(symmetric_buffer_b,
                                      symmetric_buffer_a,
                                      count,
                                      start, stride, size,
                                      (long long*)pwork,
                                      reduce_psync);
            break;
        case TAUCOLLECTIVES_SUM:
            shmem_longlong_sum_to_all(symmetric_buffer_b,
                                      symmetric_buffer_a,
                                      count,
                                      start, stride, size,
                                      (long long*)pwork,
                                      reduce_psync);
            break;
        default:
            fprintf(stderr, "Reduce: Invalid operation: %d", operation );
    }
  }
  else
  {
    fprintf(stderr, "Reduce: Invalid datatype: %d", datatype );
  }

  shmem_barrier(start, stride, size, barrier_psync);

  /* Copy symmetric memory block to buffer */
  if (rank == root)
  {
    memcpy(recvbuf, symmetric_buffer_b, count * sizeof_ipc_datatypes[ datatype ]);
  }

  shmem_barrier(start, stride, size, barrier_psync);
#endif /* TAU_SHMEM */
  return 0;
}

int TauCollectives_Allreduce(TauCollectives_Group*    group,
                             const void*              sendbuf,
                             void*                    recvbuf,
                             int                      count,
                             TauCollectives_Datatype  datatype,
                             TauCollectives_Operation operation)
{
#ifdef TAU_MPI
  return PMPI_Allreduce((void*) sendbuf,
                        recvbuf,
                        count,
                        get_mpi_datatype( datatype ),
                        get_mpi_operation( operation ),
                        group->comm);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  int start        = group->pe_start;
  int stride       = group->log_pe_stride;
  int size         = group->pe_size;
  int num_elements = count;

  if ( datatype == TAUCOLLECTIVES_BYTE ||
       datatype == TAUCOLLECTIVES_CHAR ||
       datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
  {
    num_elements = ROUNDUPTO( count, 2 );
  }

  /* Copy buffer to symmetric memory block */
  memcpy( symmetric_buffer_a, sendbuf, count * sizeof_ipc_datatypes[ datatype ] );

  shmem_barrier(start, stride, size, barrier_psync);

  /* Reduction operation */
  if(datatype == TAUCOLLECTIVES_BYTE ||
     datatype == TAUCOLLECTIVES_CHAR ||
     datatype == TAUCOLLECTIVES_UNSIGNED_CHAR )
  {
    int nreduce = CEIL( count, 2 );
    switch ( operation )
    {
      case TAUCOLLECTIVES_BAND:
        shmem_short_and_to_all(symmetric_buffer_b,
                               symmetric_buffer_a,
                               nreduce,
                               start, stride, size,
                               get_pwork(sizeof( short ), nreduce),
                               reduce_psync );
        break;
      case TAUCOLLECTIVES_BOR:
        shmem_short_or_to_all(symmetric_buffer_b,
                              symmetric_buffer_a,
                              nreduce,
                              start, stride, size,
                              get_pwork(sizeof( short ), nreduce),
                              reduce_psync);
        break;
      case TAUCOLLECTIVES_MIN:
        shmem_short_min_to_all(symmetric_buffer_b,
                               symmetric_buffer_a,
                               nreduce,
                               start, stride, size,
                               get_pwork(sizeof(short), nreduce),
                               reduce_psync);
        break;
      case TAUCOLLECTIVES_MAX:
        shmem_short_max_to_all(symmetric_buffer_b,
                               symmetric_buffer_a,
                               nreduce,
                               start, stride, size,
                               get_pwork(sizeof(short), nreduce),
                               reduce_psync);
        break;
      case TAUCOLLECTIVES_SUM:
        shmem_short_sum_to_all(symmetric_buffer_b,
                               symmetric_buffer_a,
                               nreduce,
                               start, stride, size,
                               get_pwork(sizeof( short ), nreduce),
                               reduce_psync);
        break;
      default:
        fprintf(stderr, "Allreduce: Invalid IPC operation: %d", operation );
    }
  }
  else if(datatype == TAUCOLLECTIVES_INT ||
          datatype == TAUCOLLECTIVES_UNSIGNED ||
          datatype == TAUCOLLECTIVES_INT32_T ||
          datatype == TAUCOLLECTIVES_UINT32_T )
  {
    switch ( operation )
    {
      case TAUCOLLECTIVES_BAND:
        shmem_int_and_to_all(symmetric_buffer_b,
                             symmetric_buffer_a,
                             count,
                             start, stride, size,
                             get_pwork( sizeof( int ), count ),
                             reduce_psync);
        break;
      case TAUCOLLECTIVES_BOR:
        shmem_int_or_to_all(symmetric_buffer_b,
                            symmetric_buffer_a,
                            count,
                            start, stride, size,
                            get_pwork(sizeof(int), count),
                            reduce_psync );
        break;
      case TAUCOLLECTIVES_MIN:
        shmem_int_min_to_all(symmetric_buffer_b,
                             symmetric_buffer_a,
                             count,
                             start, stride, size,
                             get_pwork(sizeof(int), count),
                             reduce_psync);
        break;
      case TAUCOLLECTIVES_MAX:
        shmem_int_max_to_all(symmetric_buffer_b,
                             symmetric_buffer_a,
                             count,
                             start, stride, size,
                             get_pwork(sizeof(int), count),
                             reduce_psync);
        break;
      case TAUCOLLECTIVES_SUM:
        shmem_int_sum_to_all(symmetric_buffer_b,
                             symmetric_buffer_a,
                             count,
                             start, stride, size,
                             get_pwork(sizeof(int), count),
                             reduce_psync);
        break;
      default:
          fprintf(stderr, "Allreduce: Invalid operation: %d", operation );
    }
  }
  else if(datatype == TAUCOLLECTIVES_INT64_T ||
          datatype == TAUCOLLECTIVES_UINT64_T ||
          datatype == TAUCOLLECTIVES_DOUBLE )
  {
    switch ( operation )
    {
      case TAUCOLLECTIVES_BAND:
        shmem_longlong_and_to_all(symmetric_buffer_b,
                                  symmetric_buffer_a,
                                  count,
                                  start, stride, size,
                                  ( long long* )pwork,
                                  reduce_psync );
        break;
      case TAUCOLLECTIVES_BOR:
        shmem_longlong_or_to_all(symmetric_buffer_b,
                                 symmetric_buffer_a,
                                 count,
                                 start, stride, size,
                                 ( long long* )pwork,
                                 reduce_psync );
        break;
      case TAUCOLLECTIVES_MIN:
        shmem_longlong_min_to_all(symmetric_buffer_b,
                                  symmetric_buffer_a,
                                  count,
                                  start, stride, size,
                                  ( long long* )pwork,
                                  reduce_psync );
        break;
      case TAUCOLLECTIVES_MAX:
        shmem_longlong_max_to_all(symmetric_buffer_b,
                                  symmetric_buffer_a,
                                  count,
                                  start, stride, size,
                                  ( long long* )pwork,
                                  reduce_psync );
        break;
      case TAUCOLLECTIVES_SUM:
        shmem_longlong_sum_to_all(symmetric_buffer_b,
                                  symmetric_buffer_a,
                                  count,
                                  start, stride, size,
                                  ( long long* )pwork,
                                  reduce_psync );
        break;
      default:
        fprintf(stderr, "Allreduce: Invalid IPC operation: %d", operation );
    }
  }
  else
  {
    fprintf(stderr, "Allreduce: Invalid IPC datatype: %d", datatype );
  }

  shmem_barrier( start, stride, size, barrier_psync );

  /* Copy symmetric memory block to buffer */
  memcpy( recvbuf, symmetric_buffer_b, count * sizeof_ipc_datatypes[ datatype ] );

  shmem_barrier(start, stride, size, barrier_psync);
#endif /* TAU_SHMEM */
  return 0;
}

int TauCollectives_Scatter(TauCollectives_Group*   group,
                           const void*             sendbuf,
                           void*                   recvbuf,
                           int                     count,
                           TauCollectives_Datatype datatype,
                           int                     root)
{
#ifdef TAU_MPI
  return PMPI_Scatter((void*) sendbuf,
                      count,
                      get_mpi_datatype( datatype ),
                      recvbuf,
                      count,
                      get_mpi_datatype( datatype ),
                      root,
                      group->comm);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  int rank         = RtsLayer::myNode();
  int start        = group->pe_start;
  int stride       = group->log_pe_stride;
  int size         = group->pe_size;

  if ( rank == root )
  {
    /* root is the only sender */

    shmem_quiet();
    for ( int receiver = start; receiver < start + size; receiver++ )
    {
        if ( receiver == root )
        {
            /* Root should not send data to itself
             * Copy data directly to buffer */
            memcpy(recvbuf,
                   &( ( ( char* )sendbuf )[ receiver * count * sizeof_ipc_datatypes[ datatype ] ] ),
                   count * sizeof_ipc_datatypes[ datatype ] );
            continue;
        }

        shmem_putmem(symmetric_buffer_b,
                     &( ( ( char* )sendbuf )[ receiver * count * sizeof_ipc_datatypes[ datatype ] ] ),
                     count * sizeof_ipc_datatypes[ datatype ],
                     receiver );
    }
    shmem_quiet();
    shmem_barrier(start, stride, size, barrier_psync);
  }
  else
  {
    /* All processing elements except root are receivers */

    /* Wait until data has arrived */
    shmem_barrier(start, stride, size, barrier_psync);

    /* Copy symmetric memory block to buffer */
    memcpy(recvbuf, symmetric_buffer_b, count * sizeof_ipc_datatypes[ datatype ]);
  }

  /* Wait until data has arrived */
  shmem_barrier( start, stride, size, barrier_psync );

#endif /* TAU_SHMEM */
  return 0;
}

int TauCollectives_Scatterv(TauCollectives_Group*   group,
                            const void*             sendbuf,
                            const int*              sendcounts,
                            void*                   recvbuf,
                            int                     recvcount,
                            TauCollectives_Datatype datatype,
                            int                     root)
{
#ifdef TAU_MPI
  int* displs = NULL;
  int  rank   = RtsLayer::myNode();
  if ( root == rank )
  {
    int size = TauCollectives_get_size(group);
    displs = (int *)calloc(size, sizeof(*displs));

    int total = 0;
    for (int i=0; i<size; i++)
    {
      displs[i] = total;
      total    += sendcounts[i];
    }
  }

  int ret =  PMPI_Scatterv((void*) sendbuf,
                           (int*) sendcounts,
                           displs,
                           get_mpi_datatype( datatype ),
                           recvbuf,
                           recvcount,
                           get_mpi_datatype( datatype ),
                           root,
                           group->comm);

  free(displs);

  return ret;
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  int rank         = RtsLayer::myNode();
  int start        = group->pe_start;
  int stride       = group->log_pe_stride;
  int size         = group->pe_size;

  if ( rank == root )
  {
    /* root is the only sender */

    int size = TauCollectives_get_size( group );

    /* Please note: at the moment SHMEM IPC groups consist of consecutive processing elements */
    CALL_SHMEM( shmem_quiet )();
    int total    = 0;
    int receiver = start;
    for ( int i = 0; i < size; i++ )
    {
        if ( receiver == root )
        {
            /* Root should not send data to itself
             * Copy data directly to buffer */
            memcpy(recvbuf,
                   &( ( ( char* )sendbuf )[ total * sizeof_ipc_datatypes[ datatype ] ] ),
                   sendcounts[ i ] * sizeof_ipc_datatypes[ datatype ] );
        }
        else
        {
            shmem_putmem(symmetric_buffer_b,
                         &( ( ( char* )sendbuf )[ total * sizeof_ipc_datatypes[ datatype ] ] ),
                         sendcounts[ i ] * sizeof_ipc_datatypes[ datatype ],
                         receiver);
        }

        total += sendcounts[i];
        receiver++;
    }
    shmem_quiet();
    shmem_barrier(start, stride, size, barrier_psync);
  }
  else
  {
    /* All processing elements except root are receivers */

    /* Wait until data has arrived */
    shmem_barrier(start, stride, size, barrier_psync);

    /* Copy symmetric memory block to buffer */
    memcpy( recvbuf, symmetric_buffer_b, recvcount * sizeof_ipc_datatypes[ datatype ] );
  }

  /* Wait until data has arrived */
  shmem_barrier(start, stride, size, barrier_psync);
#endif /* TAU_SHMEM */
  return 0;
}

TauCollectives_Group * TauCollectives_Get_World()
{
  return &tau_group_world;
}
