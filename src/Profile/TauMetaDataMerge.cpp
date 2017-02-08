/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauMetaDataMerge.c  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : MetaData merging code                            **
**                                                                         **
****************************************************************************/

#ifdef TAU_MPI
#include <mpi.h>
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
#include <shmem.h>
extern "C" void  __real_shmem_int_put(int * a1, const int * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_int_get(int * a1, const int * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_putmem(void * a1, const void * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_getmem(void * a1, const void * a2, size_t a3, int a4) ;
extern "C" int   __real_shmem_n_pes() ;
extern "C" int   __real_shmem_my_pe() ;
extern "C" void  __real_shmem_free(void * a1) ;
extern "C" void  __real_shmem_barrier_all() ;
#endif /* TAU_SHMEM */
#include <TAU.h>
#include <TauMetaData.h>
#include <TauMetrics.h>

// Moved from header file
using namespace std;


extern "C" int TAU_MPI_Finalized();


extern "C" int Tau_metadataMerge_mergeMetaData() {

  Tau_metadata_fillMetaData();

  static int merged = 0;
  if (merged == 1) {
    return 0;
  }
  merged = 1;

  int rank = 0;

#ifdef TAU_MPI
  int numRanks;
  if (TAU_MPI_Finalized()) {
    return 0;
  }

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
  int numRanks = __real_shmem_n_pes();
  rank = __real_shmem_my_pe();
  static int shBufferSize;
  int i, defBufSize;
  char *defBuf;
  Tau_util_outputDevice *out;
#endif /* TAU_SHMEM */

  x_uint64 start, end;

  if (rank == 0) {

    TAU_VERBOSE("TAU: Merging MetaData...\n");
    start = TauMetrics_getTimeOfDay();

#ifdef TAU_MPI
    Tau_util_outputDevice *out = Tau_metadata_generateMergeBuffer();
    char *defBuf = Tau_util_getOutputBuffer(out);
    int defBufSize = Tau_util_getOutputBufferLength(out);

    PMPI_Bcast(&defBufSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    PMPI_Bcast(defBuf, defBufSize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
    out = Tau_metadata_generateMergeBuffer();
    defBuf = Tau_util_getOutputBuffer(out);
    defBufSize = Tau_util_getOutputBufferLength(out) * sizeof(char);

    for(i=0; i<numRanks; i++)
      __real_shmem_int_put(&shBufferSize, &defBufSize, 1, i);
  }
  char *shBuffer = (char*)shmem_malloc((shBufferSize));
  char *Buffer = (char*)TAU_UTIL_MALLOC(shBufferSize);
  if(rank == 0) {
    for(i=0; i<shBufferSize; i++)
      shBuffer[i] = defBuf[i];
  }
  __real_shmem_barrier_all();

  __real_shmem_getmem(Buffer, shBuffer, shBufferSize, 0);
  
  if (rank == 0) {
#endif /* TAU_SHMEM */

    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: MetaData Merging Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
    char tmpstr[256];
    sprintf(tmpstr, "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU MetaData Merge Time", tmpstr);
#ifdef TAU_MPI
	Tau_util_destroyOutputDevice(out);
#endif /* TAU_MPI */
#ifdef TAU_SHMEM
	Tau_util_destroyOutputDevice(out);
#endif /* TAU_SHMEM */

  } else {
#ifdef TAU_MPI
    int BufferSize;
    PMPI_Bcast(&BufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *Buffer = (char*) TAU_UTIL_MALLOC(BufferSize);
    PMPI_Bcast(Buffer, BufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    Tau_metadata_removeDuplicates(Buffer, BufferSize);
	free(Buffer);
#endif /* TAU_MPI */
  }
#ifdef TAU_SHMEM
  __real_shmem_barrier_all();
  if(rank != 0)
    Tau_metadata_removeDuplicates(Buffer, shBufferSize);
  __real_shmem_free(shBuffer);
  free(Buffer);
#endif /* TAU_SHMEM */
  return 0;
}

