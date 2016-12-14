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
#include <TAU.h>
#include <TauMetaData.h>
#include <TauMetrics.h>

// Moved from header file
using namespace std;


extern "C" int TAU_MPI_Finalized();

extern "C" int Tau_metadataMerge_mergeMetaData_bis() {

  Tau_metadata_fillMetaData();

#if 1
  static int merged = 0;
  if (merged == 1) {
    TAU_VERBOSE("merged = 1, return\n");
    return 0;
  }
  merged = 1;
#endif

  int rank = 0;

#ifdef TAU_MPI
  int numRanks;

#if 1
  if (TAU_MPI_Finalized()) {
    TAU_VERBOSE("TAU_MPI_Finalized() called, return\n");
    return 0;
  }
#endif

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  TAU_VERBOSE("TAU Merge bis: rank=%d, numRanks=%d\n", rank, numRanks);
#endif /* TAU_MPI */

  x_uint64 start, end;

#if 1
  if (rank == 0) {

    TAU_VERBOSE("TAU: Merging MetaData...\n");
    start = TauMetrics_getTimeOfDay();

#if 1
#ifdef TAU_MPI
    Tau_util_outputDevice *out = Tau_metadata_generateMergeBuffer();
    char *defBuf = Tau_util_getOutputBuffer(out);
    int defBufSize = Tau_util_getOutputBufferLength(out);

    PMPI_Bcast(&defBufSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    PMPI_Bcast(defBuf, defBufSize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */
#endif

    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: MetaData Merging Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
    char tmpstr[256];
    sprintf(tmpstr, "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU MetaData Merge Time", tmpstr);

#if 1
#ifdef TAU_MPI
	Tau_util_destroyOutputDevice(out);
#endif /* TAU_MPI */
#endif

    TAU_VERBOSE("TAU - MetaData bis: end if condition for rank 0\n");

  } else {

#if 1
#ifdef TAU_MPI
    TAU_VERBOSE("TAU: Metadata, rank different from 0\n");
    int BufferSize;
    PMPI_Bcast(&BufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *Buffer = (char*) TAU_UTIL_MALLOC(BufferSize);
    PMPI_Bcast(Buffer, BufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    Tau_metadata_removeDuplicates(Buffer, BufferSize);
	free(Buffer);
#endif /* TAU_MPI */
#endif

    TAU_VERBOSE("TAU - MetaData bis: end if condition for other ranks\n");
  }
#endif

  TAU_VERBOSE("Tau_metadataMerge_mergeMetaData_bis END for rank #%d\n", rank);

  return 0;
}

extern "C" int Tau_metadataMerge_mergeMetaData() {

  TAU_VERBOSE("Tau_metadataMerge_mergeMetaData() begin\n");

  Tau_metadata_fillMetaData();

#if 0
  static int merged = 0;
  if (merged == 1) {
    TAU_VERBOSE("merged = 1, return\n");
    return 0;
  }
  merged = 1;
#endif

  int rank = 0;

#ifdef TAU_MPI
  int numRanks;
  if (TAU_MPI_Finalized()) {
    fprintf(stdout, "TAU_MPI_Finalized() called\n");
    return 0;
  }

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  TAU_VERBOSE("TAU: rank=%d, numRanks=%d\n", rank, numRanks);
#endif /* TAU_MPI */

  x_uint64 start, end;

#if 1
  if (rank == 0) {

    TAU_VERBOSE("TAU: Merging MetaData...\n");
    start = TauMetrics_getTimeOfDay();

#if 1
#ifdef TAU_MPI
    Tau_util_outputDevice *out = Tau_metadata_generateMergeBuffer();
    char *defBuf = Tau_util_getOutputBuffer(out);
    int defBufSize = Tau_util_getOutputBufferLength(out);

    PMPI_Bcast(&defBufSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    PMPI_Bcast(defBuf, defBufSize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */
#endif

    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: MetaData Merging Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
    char tmpstr[256];
    sprintf(tmpstr, "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU MetaData Merge Time", tmpstr);

#if 1
#ifdef TAU_MPI
	Tau_util_destroyOutputDevice(out);
#endif /* TAU_MPI */
#endif

  } else {

#if 1
#ifdef TAU_MPI
    TAU_VERBOSE("TAU: Metadata, rank different from 0\n");
    int BufferSize;
    PMPI_Bcast(&BufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *Buffer = (char*) TAU_UTIL_MALLOC(BufferSize);
    PMPI_Bcast(Buffer, BufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    Tau_metadata_removeDuplicates(Buffer, BufferSize);
	free(Buffer);
#endif /* TAU_MPI */
#endif
  }
#endif

  return 0;
}

