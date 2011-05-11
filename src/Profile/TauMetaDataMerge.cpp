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
#include <TAU.h>
#include <TauMetaData.h>
#include <TauMetrics.h>

extern "C" int TAU_MPI_Finalized();


extern "C" int Tau_metadataMerge_mergeMetaData() {

  Tau_metadata_fillMetaData();

  static int merged = 0;
  if (merged == 1) {
    return 0;
  }
  merged = 1;

  if (TAU_MPI_Finalized()) {
    return 0;
  }

  int rank, numRanks;
  MPI_Status status;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  x_uint64 start, end;

  if (rank == 0) {

    TAU_VERBOSE("TAU: Merging MetaData...\n");
    start = TauMetrics_getTimeOfDay();

    Tau_util_outputDevice *out = Tau_metadata_generateMergeBuffer();
    char *defBuf = Tau_util_getOutputBuffer(out);
    int defBufSize = Tau_util_getOutputBufferLength(out);

    PMPI_Bcast(&defBufSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    PMPI_Bcast(defBuf, defBufSize, MPI_CHAR, 0, MPI_COMM_WORLD);

    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: MetaData Merging Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
    char tmpstr[256];
    sprintf(tmpstr, "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU MetaData Merge Time", tmpstr);

  } else {
    int BufferSize;
    PMPI_Bcast(&BufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *Buffer = (char*) TAU_UTIL_MALLOC(BufferSize);
    PMPI_Bcast(Buffer, BufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    Tau_metadata_removeDuplicates(Buffer, BufferSize);
  }
}

#endif
